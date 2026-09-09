"""
Improved MNIST Real-time Recognition Demo - v2

Key improvements over v1:
  1. Smart preprocessing: bbox crop + aspect ratio + center to 28x28
     (fixes the #1 issue: tiny drawing squashed into corner of 28x28)
  2. Square 500x500 canvas (matches MNIST distribution)
  3. Thicker brush (radius 20) for cleaner strokes
  4. Smooth line interpolation (no gaps when mouse moves fast)
  5. Continuous prediction (~30 FPS while drawing)
  6. Probability smoothing (EMA over last N frames) for stable display
  7. Larger, more visible model input preview
  8. Window-size-aware brush thickness

Run: python src/realtime_opencv_v2.py
"""

import sys
import os
import time
from collections import deque

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.model import create_model
except ImportError:
    from model import create_model

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class OpenCVDemoV2:
    def __init__(self, model_path='models/simple_cnn_best.pth', canvas_size=500):
        """
        Args:
            model_path: model file path (relative to project root)
            canvas_size: square canvas side length (default 500)
        """
        # --- Paths ---
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        if not os.path.isabs(model_path):
            model_path = os.path.join(project_root, model_path)
        print(f'[INFO] Project root: {project_root}')
        print(f'[INFO] Model file: {model_path}')

        # --- Device ---
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'[INFO] Device: {self.device}')

        # --- Load model ---
        try:
            self.model = create_model('simple_cnn', num_classes=10, device=self.device)
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            accuracy = checkpoint.get('accuracy', 'N/A')
            print(f'[INFO] Model loaded (training accuracy: {accuracy}%)')
        except FileNotFoundError:
            print(f'[ERROR] Model not found: {model_path}')
            print('Run "python src/train.py" first to train the model.')
            sys.exit(1)

        # --- Canvas setup (square, matches MNIST distribution) ---
        self.canvas_size = canvas_size
        self.canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255

        # --- Drawing state ---
        self.drawing = False
        self.last_point = None
        self.base_brush_radius = 22  # thicker for clean strokes
        self.brush_radius = self.base_brush_radius

        # --- Display state ---
        self.window_width = int(canvas_size * 1.6)   # drawing area + info panel
        self.window_height = canvas_size
        self.need_update = True

        # --- Prediction smoothing (EMA over recent probabilities) ---
        self.prob_history = deque(maxlen=6)
        self.smoothed_probs = None

        # --- Trigger modes ---
        self.manual_trigger = False
        self.empty_canvas_threshold = 80  # black pixels

        # --- Throttling: continuous prediction at ~30 FPS ---
        self.last_predict_time = 0
        self.predict_interval = 1.0 / 30.0  # 33ms

        print(f'[INFO] Canvas size: {canvas_size}x{canvas_size}')
        print(f'[INFO] Brush radius: {self.base_brush_radius}')
        print(f'[INFO] Predict interval: {int(1/self.predict_interval)} FPS')

    # ====================================================================
    # Drawing: smooth line with interpolated circles
    # ====================================================================
    def draw_smooth_segment(self, img, p1, p2, radius):
        """Draw a smooth line between two points with interpolated circles.

        Avoids gaps when the mouse moves faster than the brush can fill.
        """
        p1 = np.array(p1, dtype=np.float32)
        p2 = np.array(p2, dtype=np.float32)
        dist = np.linalg.norm(p2 - p1)
        if dist < 1e-3:
            cv2.circle(img, tuple(p2.astype(int)), radius, (0, 0, 0), -1, cv2.LINE_AA)
            return
        # Interpolate every (radius / 2) pixels
        step = max(1.0, radius / 2.0)
        n_steps = max(1, int(np.ceil(dist / step)))
        for i in range(n_steps + 1):
            t = i / n_steps
            pt = (p1 + (p2 - p1) * t).astype(int)
            cv2.circle(img, tuple(pt), radius, (0, 0, 0), -1, cv2.LINE_AA)

    # ====================================================================
    # Mouse callback: handle drawing + trigger redraws
    # ====================================================================
    def mouse_callback(self, event, x, y, flags, param):
        # Compute scaled coordinates (window -> canvas)
        scale = self.window_width / self.canvas_size  # simplified; we only draw in left ~62% of window
        # Actually the left drawing area is canvas_size wide within the window.
        # Find the left drawing area boundary.
        drawing_area_w = int(self.window_width * 0.62)
        scale = self.canvas_size / drawing_area_w if drawing_area_w > 0 else 1.0

        if x >= drawing_area_w or y >= self.window_height:
            return

        cx = int(x * scale)
        cy = int(y * (self.canvas_size / self.window_height))
        cx = max(0, min(cx, self.canvas_size - 1))
        cy = max(0, min(cy, self.canvas_size - 1))

        # Adaptive brush radius based on actual canvas scale
        radius = max(6, int(self.base_brush_radius / scale))

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.last_point = (cx, cy)
            cv2.circle(self.canvas, (cx, cy), radius, (0, 0, 0), -1, cv2.LINE_AA)
            self.prob_history.clear()  # reset smoothing on new stroke
            self.need_update = True
            self.manual_trigger = False

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing and self.last_point is not None:
                self.draw_smooth_segment(self.canvas, self.last_point, (cx, cy), radius)
                self.last_point = (cx, cy)
                self.need_update = True

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.last_point = None
            self.need_update = True
            self.manual_trigger = False

    # ====================================================================
    # Core: improved preprocessing pipeline (the main fix)
    # ====================================================================
    def preprocess_canvas(self):
        """Improved preprocessing matching MNIST distribution.

        Pipeline:
          canvas (white bg, black strokes)
            -> grayscale -> binary mask
            -> bbox crop with padding
            -> square-pad (preserve aspect)
            -> resize to 20x20
            -> place at center of 28x28 black canvas
            -> normalize with MNIST mean/std
        """
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)

        # Threshold to binary mask of strokes
        _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

        # Find bounding box of strokes
        coords = cv2.findNonZero(binary)
        if coords is None:
            return None, np.zeros((28, 28), dtype=np.uint8)

        x, y, w, h = cv2.boundingRect(coords)

        # Crop with proportional padding (about 15% on each side)
        pad = int(max(w, h) * 0.15)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(self.canvas_size, x + w + pad)
        y2 = min(self.canvas_size, y + h + pad)
        cropped = binary[y1:y2, x1:x2]

        # Square-pad preserving aspect ratio
        ch, cw = cropped.shape
        side = max(ch, cw)
        top = (side - ch) // 2
        bot = side - ch - top
        left = (side - cw) // 2
        right = side - cw - left
        square = cv2.copyMakeBorder(
            cropped, top, bot, left, right,
            cv2.BORDER_CONSTANT, value=0
        )

        # Resize to 20x20 (MNIST digits are ~20x20 within the 28x28 frame)
        digit = cv2.resize(square, (20, 20), interpolation=cv2.INTER_AREA)

        # Place at center of 28x28 black canvas (4-pixel border on each side)
        canvas28 = np.zeros((28, 28), dtype=np.uint8)
        canvas28[4:24, 4:24] = digit

        # Normalize to match MNIST training distribution
        normalized = canvas28.astype(np.float32) / 255.0
        normalized = (normalized - 0.1307) / 0.3081

        tensor = torch.FloatTensor(normalized).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device), canvas28

    # ====================================================================
    # Prediction with EMA smoothing
    # ====================================================================
    def predict(self):
        """Run model inference, return smoothed probabilities + raw probs."""
        tensor, processed = self.preprocess_canvas()
        if tensor is None:
            # Empty canvas -> force predict 0 with 100% confidence
            self.smoothed_probs = torch.zeros(10, device=self.device)
            self.smoothed_probs[0] = 1.0
            return self.smoothed_probs.cpu(), 0, processed

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = F.softmax(outputs, dim=1)[0]

        # EMA smoothing
        self.prob_history.append(probs)
        if len(self.prob_history) > 0:
            self.smoothed_probs = torch.stack(list(self.prob_history)).mean(dim=0)
        else:
            self.smoothed_probs = probs

        predicted = int(self.smoothed_probs.argmax().item())
        return self.smoothed_probs.cpu(), predicted, processed

    # ====================================================================
    # Render UI: layout the canvas + info panel
    # ====================================================================
    def render_ui(self, predicted, probabilities, processed_img, is_empty):
        """Build the full display frame."""
        # Drawing area: left 62% of window, square aspect
        drawing_w = int(self.window_width * 0.62)
        drawing_w = min(drawing_w, self.window_height)
        drawing_h = self.window_height

        # Scale canvas to drawing area (preserving square)
        display_canvas = cv2.resize(self.canvas, (drawing_w, drawing_h),
                                    interpolation=cv2.INTER_AREA)
        # Add light grid for visual reference (subtle)
        grid_color = (240, 240, 240)
        for i in range(1, 4):
            y = i * drawing_h // 4
            cv2.line(display_canvas, (0, y), (drawing_w, y), grid_color, 1)

        # Info panel: right 38%
        panel_w = self.window_width - drawing_w
        panel = np.ones((self.window_height, panel_w, 3), dtype=np.uint8) * 245

        # Dynamic layout based on panel width
        margin = max(15, panel_w // 30)
        bar_max_w = panel_w - 3 * margin

        # ---- Section A: Title ----
        title_y = 50
        if is_empty:
            title = f'Prediction: 0 (Empty)'
            color = (0, 0, 200)
        elif self.manual_trigger:
            title = f'Prediction: {predicted} (Manual)'
            color = (200, 0, 0)
        else:
            title = f'Prediction: {predicted}'
            color = (0, 130, 0)

        scale_w = panel_w / 400.0
        font_scale = max(0.7, min(1.4, 0.9 * scale_w))
        thickness = max(2, int(2 * scale_w))
        (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.putText(panel, title,
                    ((panel_w - tw) // 2, title_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

        # ---- Section B: Confidence bars ----
        bar_section_y = 90
        bar_height = max(15, int(panel_h_unit := (self.window_height - 320) // 11))
        bar_spacing = bar_height + max(4, bar_height // 4)
        for i in range(10):
            prob = probabilities[i].item() * 100
            bar_w = int(prob * bar_max_w / 100.0)
            y = bar_section_y + i * bar_spacing

            # Background bar
            cv2.rectangle(panel, (margin, y),
                          (margin + bar_max_w, y + bar_height),
                          (220, 220, 220), -1)
            # Filled bar (highlighted if this is prediction)
            if bar_w > 0:
                bar_color = (60, 180, 60) if i == predicted else (160, 160, 160)
                cv2.rectangle(panel, (margin, y),
                              (margin + bar_w, y + bar_height),
                              bar_color, -1)
            # Label
            cv2.putText(panel, f'{i}',
                        (margin - 5 if margin > 18 else 2, y + bar_height - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, (0, 0, 0), 1, cv2.LINE_AA)
            # Percentage
            pct_text = f'{prob:.1f}%'
            cv2.putText(panel, pct_text,
                        (margin + bar_max_w + 5, y + bar_height - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, (50, 50, 50), 1, cv2.LINE_AA)

        # ---- Section C: Model input preview ----
        preview_size = max(80, min(panel_w - 2 * margin, 160))
        preview_y = self.window_height - preview_size - 35
        preview_x = (panel_w - preview_size) // 2
        # White border
        cv2.rectangle(panel,
                      (preview_x - 2, preview_y - 2),
                      (preview_x + preview_size + 2, preview_y + preview_size + 2),
                      (255, 255, 255), -1)
        # Black border
        cv2.rectangle(panel,
                      (preview_x - 2, preview_y - 2),
                      (preview_x + preview_size + 2, preview_y + preview_size + 2),
                      (100, 100, 100), 1)
        # The 28x28 image
        small = cv2.resize(processed_img, (preview_size, preview_size),
                           interpolation=cv2.INTER_NEAREST)
        small_bgr = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
        panel[preview_y:preview_y + preview_size,
              preview_x:preview_x + preview_size] = small_bgr
        # Label
        cv2.putText(panel, 'Model Input (28x28)',
                    (preview_x, preview_y + preview_size + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.6, (100, 100, 100), 1, cv2.LINE_AA)

        # Combine
        combined = np.hstack([display_canvas, panel])
        return combined

    # ====================================================================
    # Main loop
    # ====================================================================
    def run(self):
        window_name = 'MNIST Real-time Recognition v2 (Improved)'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.window_width, self.window_height)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        print('\n' + '=' * 64)
        print(' MNIST Real-time Recognition v2 (Improved)')
        print('=' * 64)
        print(' Controls:')
        print('   Left-mouse drag    - draw a digit')
        print('   c                  - clear canvas')
        print('   s                  - manual predict (prints all probs)')
        print('   + / -              - brush thicker / thinner')
        print('   r                  - reset smoothing')
        print('   ESC                - quit')
        print()
        print(' Improvements:')
        print('   * Smart preprocessing (bbox + aspect + center)')
        print('   * Continuous ~30 FPS prediction while drawing')
        print('   * EMA smoothing for stable predictions')
        print('   * Smooth interpolated strokes')
        print('=' * 64 + '\n')

        # Track key state for debouncing
        last_key_state = {}

        while True:
            # Detect window resize
            try:
                rect = cv2.getWindowImageRect(window_name)
                if rect and len(rect) == 4 and rect[2] > 50 and rect[3] > 50:
                    if (abs(rect[2] - self.window_width) > 2 or
                            abs(rect[3] - self.window_height) > 2):
                        self.window_width = rect[2]
                        self.window_height = rect[3]
                        self.need_update = True
            except Exception:
                pass

            # Throttled continuous prediction
            now = time.time()
            should_predict = (
                self.need_update and
                (now - self.last_predict_time) >= self.predict_interval
            )

            if should_predict:
                # Detect empty canvas
                gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
                black_pixels = int(np.sum(gray < 80))
                is_empty = black_pixels < self.empty_canvas_threshold

                if is_empty:
                    self.smoothed_probs = torch.zeros(10)
                    self.smoothed_probs[0] = 1.0
                    self.prob_history.clear()
                    probabilities = self.smoothed_probs
                    predicted = 0
                    processed_img = np.zeros((28, 28), dtype=np.uint8)
                else:
                    probabilities, predicted, processed_img = self.predict()

                # If user just finished drawing, reset smoothing
                # (so next stroke starts fresh)
                frame = self.render_ui(predicted, probabilities, processed_img, is_empty)
                cv2.imshow(window_name, frame)
                self.need_update = False
                self.last_predict_time = now

                # Print probs on manual trigger
                if self.manual_trigger:
                    print(f'[Manual] prediction={predicted}, black_pixels={black_pixels}')
                    if not is_empty:
                        for i, p in enumerate(probabilities):
                            print(f'  {i}: {p.item() * 100:5.2f}%')
                    print('-' * 40)

            # Keyboard
            key = cv2.waitKey(20) & 0xFF

            if key == 27:  # ESC
                print('[INFO] ESC pressed, exiting.')
                break
            elif key == ord('c'):
                self.canvas = np.ones((self.canvas_size, self.canvas_size, 3), dtype=np.uint8) * 255
                self.prob_history.clear()
                self.need_update = True
                self.manual_trigger = False
                print('[INFO] Canvas cleared.')
            elif key == ord('s'):
                if not last_key_state.get('s', False):
                    self.need_update = True
                    self.manual_trigger = True
                    print('[INFO] Manual trigger.')
                last_key_state['s'] = True
            else:
                last_key_state['s'] = False

            # Brush adjustment
            if key == ord('+') or key == ord('='):
                self.base_brush_radius = min(60, self.base_brush_radius + 2)
                print(f'[INFO] Brush radius: {self.base_brush_radius}')
                self.need_update = True
            elif key == ord('-') or key == ord('_'):
                self.base_brush_radius = max(4, self.base_brush_radius - 2)
                print(f'[INFO] Brush radius: {self.base_brush_radius}')
                self.need_update = True
            elif key == ord('r'):
                self.prob_history.clear()
                self.need_update = True
                print('[INFO] Smoothing reset.')

        cv2.destroyAllWindows()
        print('[INFO] Goodbye.')


if __name__ == '__main__':
    try:
        demo = OpenCVDemoV2(model_path='models/simple_cnn_best.pth', canvas_size=500)
        demo.run()
    except KeyboardInterrupt:
        print('\n[INFO] Interrupted.')
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
