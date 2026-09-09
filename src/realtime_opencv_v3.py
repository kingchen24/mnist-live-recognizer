"""
MNIST Real-time Recognition Demo - v3

Dual-mode demo with both mouse drawing and camera capture.

Key improvements over v2:
  - Press 'v' to toggle between mouse mode and camera mode
  - Camera mode: live webcam feed with auto ROI detection
  - Adaptive thresholding for varying lighting
  - Largest contour heuristic for digit detection
  - ROI size adjustable with [ / ]
  - Confidence threshold to suppress low-confidence noise

Run: python src/realtime_opencv_v3.py
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


class OpenCVDemoV3:
    """Dual-mode MNIST recognition demo."""

    # Mode constants
    MODE_MOUSE = 'mouse'
    MODE_CAMERA = 'camera'

    def __init__(self, model_path='models/simple_cnn_best.pth', canvas_size=500,
                 camera_id=0, frame_w=640, frame_h=480):
        # --- Paths ---
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
            print(f'[INFO] Model loaded (training accuracy: '
                  f'{checkpoint.get("accuracy", "N/A")}%)')
        except FileNotFoundError:
            print(f'[ERROR] Model not found: {model_path}')
            sys.exit(1)

        # --- Mouse mode setup ---
        self.canvas_size = canvas_size
        self.canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255
        self.base_brush_radius = 22

        # --- Camera setup ---
        self.camera_id = camera_id
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.cap = None
        self.roi_size = 240  # square ROI size
        self.roi_offset = (0, 0)  # ROI offset from center

        # --- Mode state ---
        self.mode = self.MODE_MOUSE

        # --- Display state ---
        self.window_width = int(canvas_size * 1.6)
        self.window_height = canvas_size
        self.need_update = True

        # --- Prediction smoothing ---
        self.prob_history = deque(maxlen=6)
        self.smoothed_probs = None

        # --- Triggers ---
        self.manual_trigger = False
        self.empty_canvas_threshold = 80
        self.min_confidence = 0.0  # below this, hide prediction

        # --- Throttling ---
        self.last_predict_time = 0
        self.predict_interval = 1.0 / 30.0
        self.last_camera_read = 0
        self.camera_interval = 1.0 / 25.0  # 25 FPS from camera

        print(f'[INFO] Canvas size: {canvas_size}x{canvas_size}')
        print(f'[INFO] Mode: {self.mode}')

    # ====================================================================
    # Mode switching
    # ====================================================================
    def switch_mode(self):
        """Toggle between mouse and camera modes."""
        if self.mode == self.MODE_MOUSE:
            self._enter_camera_mode()
        else:
            self._enter_mouse_mode()

    def _enter_camera_mode(self):
        print('\n[MODE] Switching to camera...')
        # Try with CAP_DSHOW first (better Windows support), fallback to default
        self.cap = None
        for backend in [cv2.CAP_DSHOW, cv2.CAP_ANY]:
            try:
                cap = cv2.VideoCapture(self.camera_id, backend)
                if cap.isOpened():
                    self.cap = cap
                    print(f'[INFO] Opened with backend: {backend}')
                    break
                cap.release()
            except Exception:
                pass
        if self.cap is None:
            print(f'[ERROR] Cannot open camera {self.camera_id}')
            print('Possible causes:')
            print('  1. No camera connected')
            print('  2. Camera in use by another app (close Zoom/Teams/etc.)')
            print('  3. Permission denied (Settings > Privacy > Camera)')
            print(f'  4. Wrong camera index (try --camera-id 1 or 2)')
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_h)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Warm-up: read a few frames to let auto-exposure settle
        for _ in range(5):
            self.cap.read()
        self.mode = self.MODE_CAMERA
        self.prob_history.clear()
        self.need_update = True
        print(f'[MODE] Camera active ({self.frame_w}x{self.frame_h}). '
              f'Hold a hand-written digit in the green box.')
        return True

    def _enter_mouse_mode(self):
        print('\n[MODE] Switching to mouse...')
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.mode = self.MODE_MOUSE
        self.prob_history.clear()
        self.need_update = True

    # ====================================================================
    # Mouse drawing (mouse mode only)
    # ====================================================================
    def draw_smooth_segment(self, img, p1, p2, radius):
        p1 = np.array(p1, dtype=np.float32)
        p2 = np.array(p2, dtype=np.float32)
        dist = np.linalg.norm(p2 - p1)
        if dist < 1e-3:
            cv2.circle(img, tuple(p2.astype(int)), radius, (0, 0, 0), -1, cv2.LINE_AA)
            return
        step = max(1.0, radius / 2.0)
        n_steps = max(1, int(np.ceil(dist / step)))
        for i in range(n_steps + 1):
            t = i / n_steps
            pt = (p1 + (p2 - p1) * t).astype(int)
            cv2.circle(img, tuple(pt), radius, (0, 0, 0), -1, cv2.LINE_AA)

    def mouse_callback(self, event, x, y, flags, param):
        """Mouse handler for mouse mode (drawing)."""
        if self.mode != self.MODE_MOUSE:
            return

        drawing_area_w = int(self.window_width * 0.62)
        scale = self.canvas_size / drawing_area_w if drawing_area_w > 0 else 1.0

        if x >= drawing_area_w or y >= self.window_height:
            return

        cx = int(x * scale)
        cy = int(y * (self.canvas_size / self.window_height))
        cx = max(0, min(cx, self.canvas_size - 1))
        cy = max(0, min(cy, self.canvas_size - 1))

        radius = max(6, int(self.base_brush_radius / scale))

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.last_point = (cx, cy)
            cv2.circle(self.canvas, (cx, cy), radius, (0, 0, 0), -1, cv2.LINE_AA)
            self.prob_history.clear()
            self.need_update = True
            self.manual_trigger = False
        elif event == cv2.EVENT_MOUSEMOVE:
            if getattr(self, 'drawing', False) and getattr(self, 'last_point', None):
                self.draw_smooth_segment(self.canvas, self.last_point, (cx, cy), radius)
                self.last_point = (cx, cy)
                self.need_update = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.last_point = None
            self.need_update = True
            self.manual_trigger = False

    # ====================================================================
    # Preprocessing pipelines
    # ====================================================================
    def preprocess_canvas(self):
        """Preprocess mouse-mode canvas."""
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        coords = cv2.findNonZero(binary)
        if coords is None:
            return None, np.zeros((28, 28), dtype=np.uint8)
        return self._crop_and_normalize(binary, coords)

    def preprocess_camera_frame(self, frame):
        """Preprocess camera frame ROI into 28x28 model input.

        Steps:
          1. Extract ROI from camera frame
          2. Convert to grayscale
          3. Adaptive threshold (handles varying lighting)
          4. Find largest contour (likely the digit)
          5. Crop, square-pad, resize to 20x20
          6. Place in 28x28 black canvas
        """
        x0, y0, w, h = self._get_roi()
        roi = frame[y0:y0 + h, x0:x0 + w]
        if roi.size == 0:
            return None, np.zeros((28, 28), dtype=np.uint8)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Blur slightly to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Find the largest contour (most likely the digit)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, np.zeros((28, 28), dtype=np.uint8)
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        # Filter: digit must occupy at least 5% of ROI
        if area < w * h * 0.02:
            return None, np.zeros((28, 28), dtype=np.uint8)
        coords = cv2.boundingRect(largest)
        # boundingRect returns (x, y, w, h); findNonZero needs contour-like
        x, y, cw, ch = coords
        # Build a single contour from the bbox (for compatibility)
        contour_pts = np.array([
            [[x, y]], [[x + cw, y]], [[x + cw, y + ch]], [[x, y + ch]]
        ], dtype=np.int32)
        return self._crop_and_normalize(binary, contour_pts)

    def _crop_and_normalize(self, binary, coords):
        """Crop bbox, square-pad, resize to 20x20, place in 28x28 black canvas."""
        x, y, w, h = cv2.boundingRect(coords)
        pad = int(max(w, h) * 0.15)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(binary.shape[1], x + w + pad)
        y2 = min(binary.shape[0], y + h + pad)
        cropped = binary[y1:y2, x1:x2]
        ch, cw = cropped.shape
        side = max(ch, cw)
        top = (side - ch) // 2
        bot = side - ch - top
        left = (side - cw) // 2
        right = side - cw - left
        square = cv2.copyMakeBorder(
            cropped, top, bot, left, right, cv2.BORDER_CONSTANT, value=0
        )
        digit = cv2.resize(square, (20, 20), interpolation=cv2.INTER_AREA)
        canvas28 = np.zeros((28, 28), dtype=np.uint8)
        canvas28[4:24, 4:24] = digit
        normalized = canvas28.astype(np.float32) / 255.0
        normalized = (normalized - 0.1307) / 0.3081
        tensor = torch.FloatTensor(normalized).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device), canvas28

    # ====================================================================
    # Camera helpers
    # ====================================================================
    def _get_roi(self):
        """Compute ROI (x, y, w, h) in camera frame."""
        cx = self.frame_w // 2 + self.roi_offset[0]
        cy = self.frame_h // 2 + self.roi_offset[1]
        s = self.roi_size
        x = max(0, min(cx - s // 2, self.frame_w - s))
        y = max(0, min(cy - s // 2, self.frame_h - s))
        return x, y, s, s

    # ====================================================================
    # Prediction with EMA smoothing
    # ====================================================================
    def predict(self, tensor, processed_img):
        if tensor is None:
            self.smoothed_probs = torch.zeros(10)
            self.smoothed_probs[0] = 1.0
            return self.smoothed_probs, 0, processed_img
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = F.softmax(outputs, dim=1)[0]
        self.prob_history.append(probs)
        if len(self.prob_history) > 0:
            self.smoothed_probs = torch.stack(list(self.prob_history)).mean(dim=0)
        else:
            self.smoothed_probs = probs
        predicted = int(self.smoothed_probs.argmax().item())
        return self.smoothed_probs.cpu(), predicted, processed_img

    # ====================================================================
    # UI rendering
    # ====================================================================
    def render_mouse_ui(self, predicted, probabilities, processed_img, is_empty):
        """Render mouse-mode UI."""
        drawing_w = int(self.window_width * 0.62)
        drawing_w = min(drawing_w, self.window_height)
        drawing_h = self.window_height
        display_canvas = cv2.resize(self.canvas, (drawing_w, drawing_h),
                                    interpolation=cv2.INTER_AREA)
        # Subtle grid
        grid_color = (240, 240, 240)
        for i in range(1, 4):
            y = i * drawing_h // 4
            cv2.line(display_canvas, (0, y), (drawing_w, y), grid_color, 1)

        panel = self._build_info_panel(
            predicted, probabilities, processed_img, is_empty,
            width=self.window_width - drawing_w,
            height=self.window_height,
            show_confidence=not is_empty
        )
        return np.hstack([display_canvas, panel])

    def render_camera_ui(self, frame, predicted, probabilities,
                          processed_img, is_empty):
        """Render camera-mode UI."""
        # Resize camera frame to fit drawing area
        drawing_w = int(self.window_width * 0.62)
        drawing_w = min(drawing_w, self.window_height)
        drawing_h = self.window_height

        scale = min(drawing_w / self.frame_w, drawing_h / self.frame_h)
        new_w = int(self.frame_w * scale)
        new_h = int(self.frame_h * scale)
        display = cv2.resize(frame, (new_w, new_h))

        # Pad to drawing area size
        padded = np.ones((drawing_h, drawing_w, 3), dtype=np.uint8) * 40
        x_off = (drawing_w - new_w) // 2
        y_off = (drawing_h - new_h) // 2
        padded[y_off:y_off + new_h, x_off:x_off + new_w] = display

        # Draw ROI rectangle on padded image (map ROI coords to display coords)
        rx, ry, rw, rh = self._get_roi()
        rx_d = int(rx * scale) + x_off
        ry_d = int(ry * scale) + y_off
        rw_d = int(rw * scale)
        rh_d = int(rh * scale)
        color_roi = (0, 255, 0) if not is_empty else (0, 200, 200)
        cv2.rectangle(padded, (rx_d, ry_d),
                      (rx_d + rw_d, ry_d + rh_d), color_roi, 2)
        # Corner markers for the ROI
        corner_len = 18
        for cx, cy, dx, dy in [
            (rx_d, ry_d, 1, 1),
            (rx_d + rw_d, ry_d, -1, 1),
            (rx_d, ry_d + rh_d, 1, -1),
            (rx_d + rw_d, ry_d + rh_d, -1, -1),
        ]:
            cv2.line(padded, (cx, cy), (cx + dx * corner_len, cy), color_roi, 3)
            cv2.line(padded, (cx, cy), (cx, cy + dy * corner_len), color_roi, 3)

        # Mode label
        cv2.putText(padded, 'CAMERA MODE', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(padded, 'Position digit inside green box',
                    (10, drawing_h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        panel = self._build_info_panel(
            predicted, probabilities, processed_img, is_empty,
            width=self.window_width - drawing_w,
            height=self.window_height,
            show_confidence=not is_empty,
            mode='camera'
        )
        return np.hstack([padded, panel])

    def _build_info_panel(self, predicted, probabilities, processed_img,
                          is_empty, width, height, show_confidence=True,
                          mode='mouse'):
        """Build the right-side info panel (shared by both modes)."""
        panel = np.ones((height, width, 3), dtype=np.uint8) * 245
        margin = max(15, width // 30)
        bar_max_w = width - 3 * margin
        scale_w = width / 400.0
        font_scale = max(0.7, min(1.4, 0.9 * scale_w))
        thickness = max(2, int(2 * scale_w))

        # Section A: Title
        if is_empty:
            title = 'No digit detected'
            color = (0, 0, 200)
        else:
            title = f'Prediction: {predicted}'
            color = (0, 130, 0)
        (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX,
                                       font_scale, thickness)
        cv2.putText(panel, title, ((width - tw) // 2, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness,
                    cv2.LINE_AA)

        # Section B: Confidence bars
        bar_section_y = 90
        bar_height = max(15, int((height - 320) // 11))
        bar_spacing = bar_height + max(4, bar_height // 4)
        for i in range(10):
            prob = probabilities[i].item() * 100
            bar_w = int(prob * bar_max_w / 100.0)
            y = bar_section_y + i * bar_spacing
            cv2.rectangle(panel, (margin, y),
                          (margin + bar_max_w, y + bar_height),
                          (220, 220, 220), -1)
            if bar_w > 0:
                bar_color = (60, 180, 60) if i == predicted else (160, 160, 160)
                cv2.rectangle(panel, (margin, y),
                              (margin + bar_w, y + bar_height),
                              bar_color, -1)
            cv2.putText(panel, f'{i}',
                        (max(2, margin - 18), y + bar_height - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7,
                        (0, 0, 0), 1, cv2.LINE_AA)
            pct_text = f'{prob:.1f}%'
            cv2.putText(panel, pct_text,
                        (margin + bar_max_w + 5, y + bar_height - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7,
                        (50, 50, 50), 1, cv2.LINE_AA)

        # Section C: 28x28 preview
        preview_size = max(80, min(width - 2 * margin, 160))
        preview_y = height - preview_size - 35
        preview_x = (width - preview_size) // 2
        cv2.rectangle(panel,
                      (preview_x - 2, preview_y - 2),
                      (preview_x + preview_size + 2, preview_y + preview_size + 2),
                      (255, 255, 255), -1)
        cv2.rectangle(panel,
                      (preview_x - 2, preview_y - 2),
                      (preview_x + preview_size + 2, preview_y + preview_size + 2),
                      (100, 100, 100), 1)
        small = cv2.resize(processed_img, (preview_size, preview_size),
                           interpolation=cv2.INTER_NEAREST)
        small_bgr = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
        panel[preview_y:preview_y + preview_size,
              preview_x:preview_x + preview_size] = small_bgr
        cv2.putText(panel, 'Model Input (28x28)',
                    (preview_x, preview_y + preview_size + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.6,
                    (100, 100, 100), 1, cv2.LINE_AA)
        return panel

    # ====================================================================
    # Main loop
    # ====================================================================
    def run(self):
        window_name = 'MNIST Real-time Recognition v3 (Mouse + Camera)'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.window_width, self.window_height)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        print('\n' + '=' * 68)
        print(' MNIST Real-time Recognition v3 (Mouse + Camera)')
        print('=' * 68)
        print(' Controls:')
        print('   v               - toggle between MOUSE and CAMERA mode')
        print('   Left-mouse drag - draw a digit (mouse mode only)')
        print('   c               - clear canvas (mouse mode)')
        print('   s               - manual predict (prints all probs)')
        print('   + / -           - brush thicker / thinner (mouse mode)')
        print('   [ / ]           - ROI smaller / larger (camera mode)')
        print('   Arrow keys      - move ROI (camera mode)')
        print('   r               - reset smoothing / ROI position')
        print('   ESC             - quit')
        print()
        print(' Tips for camera mode:')
        print('   - Hold a hand-written digit in the green box')
        print('   - Use a dark marker on white paper for best results')
        print('   - Avoid shadows and uneven lighting')
        print('=' * 68 + '\n')

        last_key_state = {}
        while True:
            # Handle window resize
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

            now = time.time()
            should_predict = (
                self.need_update and
                (now - self.last_predict_time) >= self.predict_interval
            )

            if should_predict:
                if self.mode == self.MODE_MOUSE:
                    self._process_mouse_frame()
                else:
                    self._process_camera_frame()
                self.last_predict_time = now
                self.need_update = False

            # Keyboard
            key = cv2.waitKey(20) & 0xFF

            if key == 27:  # ESC
                print('[INFO] ESC pressed, exiting.')
                break
            elif key == ord('v'):
                self.switch_mode()
            elif key == ord('c') and self.mode == self.MODE_MOUSE:
                self.canvas = np.ones(
                    (self.canvas_size, self.canvas_size, 3), dtype=np.uint8
                ) * 255
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

            # Mouse mode: brush adjustment
            if self.mode == self.MODE_MOUSE:
                if key == ord('+') or key == ord('='):
                    self.base_brush_radius = min(60, self.base_brush_radius + 2)
                    self.need_update = True
                elif key == ord('-') or key == ord('_'):
                    self.base_brush_radius = max(4, self.base_brush_radius - 2)
                    self.need_update = True
            # Camera mode: ROI adjustment
            elif self.mode == self.MODE_CAMERA:
                if key == ord(']') or key == ord('}'):
                    self.roi_size = min(480, self.roi_size + 20)
                    self.need_update = True
                    print(f'[INFO] ROI size: {self.roi_size}')
                elif key == ord('[') or key == ord('{'):
                    self.roi_size = max(80, self.roi_size - 20)
                    self.need_update = True
                    print(f'[INFO] ROI size: {self.roi_size}')
                # Arrow keys
                elif key == 82:  # Up
                    self.roi_offset = (self.roi_offset[0], self.roi_offset[1] - 20)
                    self.need_update = True
                elif key == 84:  # Down
                    self.roi_offset = (self.roi_offset[0], self.roi_offset[1] + 20)
                    self.need_update = True
                elif key == 81:  # Left
                    self.roi_offset = (self.roi_offset[0] - 20, self.roi_offset[1])
                    self.need_update = True
                elif key == 83:  # Right
                    self.roi_offset = (self.roi_offset[0] + 20, self.roi_offset[1])
                    self.need_update = True

            if key == ord('r'):
                self.prob_history.clear()
                self.roi_offset = (0, 0)
                self.need_update = True
                print('[INFO] Reset.')

        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print('[INFO] Goodbye.')

    # ====================================================================
    # Frame processors (one per mode)
    # ====================================================================
    def _process_mouse_frame(self):
        """Mouse mode: predict on canvas, render UI."""
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
            tensor, processed_img = self.preprocess_canvas()
            probabilities, predicted, processed_img = self.predict(
                tensor, processed_img
            )

        frame = self.render_mouse_ui(
            predicted, probabilities, processed_img, is_empty
        )
        cv2.imshow('MNIST Real-time Recognition v3 (Mouse + Camera)', frame)

        if self.manual_trigger:
            print(f'[Manual] pred={predicted}, black_pixels={black_pixels}')
            for i, p in enumerate(probabilities):
                print(f'  {i}: {p.item() * 100:5.2f}%')
            print('-' * 40)
            self.manual_trigger = False

    def _process_camera_frame(self):
        """Camera mode: capture frame, predict, render UI."""
        if self.cap is None or not self.cap.isOpened():
            return
        ret, frame = self.cap.read()
        if not ret:
            return

        tensor, processed_img = self.preprocess_camera_frame(frame)
        is_empty = tensor is None
        if is_empty:
            self.smoothed_probs = torch.zeros(10)
            self.smoothed_probs[0] = 1.0
            self.prob_history.clear()
            probabilities = self.smoothed_probs
            predicted = 0
        else:
            probabilities, predicted, processed_img = self.predict(
                tensor, processed_img
            )

        frame_ui = self.render_camera_ui(
            frame, predicted, probabilities, processed_img, is_empty
        )
        cv2.imshow('MNIST Real-time Recognition v3 (Mouse + Camera)', frame_ui)

        if self.manual_trigger:
            print(f'[Manual] pred={predicted}, empty={is_empty}')
            for i, p in enumerate(probabilities):
                print(f'  {i}: {p.item() * 100:5.2f}%')
            print('-' * 40)
            self.manual_trigger = False


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='MNIST Real-time Recognition v3')
    parser.add_argument('--camera-id', type=int, default=0,
                        help='Camera index (default 0; try 1 or 2 if you have multiple cameras)')
    parser.add_argument('--frame-w', type=int, default=640,
                        help='Camera frame width (default 640)')
    parser.add_argument('--frame-h', type=int, default=480,
                        help='Camera frame height (default 480)')
    args = parser.parse_args()
    try:
        demo = OpenCVDemoV3(
            model_path='models/simple_cnn_best.pth',
            canvas_size=500,
            camera_id=args.camera_id,
            frame_w=args.frame_w,
            frame_h=args.frame_h
        )
        demo.run()
    except KeyboardInterrupt:
        print('\n[INFO] Interrupted.')
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)


