"""
MNIST Real-time Recognition Demo - v2.5

Improvements over v2:
  - On-screen Clear button (no keyboard needed)
  - On-screen brush size [-] [size] [+] buttons
  - Visual brush size indicator
  - Fixed layout: percentages no longer cut off
  - Button hover state with color change

Run: python src/realtime_opencv_v2_5.py
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


class OpenCVDemoV25:
    # UI button states
    BTN_HOVER = 'hover'
    BTN_NORMAL = 'normal'

    # Color palette
    COLOR_BG = (245, 245, 245)
    COLOR_BTN = (220, 220, 225)
    COLOR_BTN_HOVER = (180, 200, 240)
    COLOR_BTN_ACTIVE = (120, 160, 220)
    COLOR_TEXT = (40, 40, 40)
    COLOR_ACCENT = (60, 60, 60)

    def __init__(self, model_path='models/simple_cnn_best.pth', canvas_size=500):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        if not os.path.isabs(model_path):
            model_path = os.path.join(project_root, model_path)
        print(f'[INFO] Project root: {project_root}')
        print(f'[INFO] Model file: {model_path}')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'[INFO] Device: {self.device}')

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

        self.canvas_size = canvas_size
        self.canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255
        self.base_brush_radius = 22
        self.min_brush = 6
        self.max_brush = 60

        self.drawing = False
        self.last_point = None
        self.prob_history = deque(maxlen=6)
        self.smoothed_probs = None
        self.manual_trigger = False
        self.empty_canvas_threshold = 80
        self.last_predict_time = 0
        self.predict_interval = 1.0 / 30.0

        # Window
        self.window_width = int(canvas_size * 1.6)
        self.window_height = int(canvas_size * 1.2)  # taller for buttons
        self.need_update = True

        # Buttons (computed each frame in render_ui)
        self.buttons = {}  # name -> (x1, y1, x2, y2)
        self.hovered_button = None

        print(f'[INFO] Canvas: {canvas_size}x{canvas_size}, brush: {self.base_brush_radius}')

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

    def preprocess_canvas(self):
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        coords = cv2.findNonZero(binary)
        if coords is None:
            return None, np.zeros((28, 28), dtype=np.uint8)
        x, y, w, h = cv2.boundingRect(coords)
        pad = int(max(w, h) * 0.15)
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(self.canvas_size, x + w + pad), min(self.canvas_size, y + h + pad)
        cropped = binary[y1:y2, x1:x2]
        ch, cw = cropped.shape
        side = max(ch, cw)
        top = (side - ch) // 2
        bot = side - ch - top
        left = (side - cw) // 2
        right = side - cw - left
        square = cv2.copyMakeBorder(cropped, top, bot, left, right,
                                    cv2.BORDER_CONSTANT, value=0)
        digit = cv2.resize(square, (20, 20), interpolation=cv2.INTER_AREA)
        canvas28 = np.zeros((28, 28), dtype=np.uint8)
        canvas28[4:24, 4:24] = digit
        normalized = canvas28.astype(np.float32) / 255.0
        normalized = (normalized - 0.1307) / 0.3081
        tensor = torch.FloatTensor(normalized).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device), canvas28

    def predict(self):
        tensor, processed = self.preprocess_canvas()
        if tensor is None:
            self.smoothed_probs = torch.zeros(10)
            self.smoothed_probs[0] = 1.0
            return self.smoothed_probs.cpu(), 0, processed
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = F.softmax(outputs, dim=1)[0]
        self.prob_history.append(probs)
        if len(self.prob_history) > 0:
            self.smoothed_probs = torch.stack(list(self.prob_history)).mean(dim=0)
        else:
            self.smoothed_probs = probs
        predicted = int(self.smoothed_probs.argmax().item())
        return self.smoothed_probs.cpu(), predicted, processed

    # ====================================================================
    # Button drawing
    # ====================================================================
    def draw_button(self, img, x1, y1, x2, y2, label, state='normal'):
        """Draw a button with label and current state."""
        if state == 'active':
            color = self.COLOR_BTN_ACTIVE
        elif state == 'hover':
            color = self.COLOR_BTN_HOVER
        else:
            color = self.COLOR_BTN
        # Background with rounded corners (simulated by rectangle)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        # Border
        cv2.rectangle(img, (x1, y1), (x2, y2), self.COLOR_ACCENT, 2)
        # Label centered
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, min(1.0, (y2 - y1) / 40.0))
        thickness = 2
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        tx = x1 + (x2 - x1 - tw) // 2
        ty = y1 + (y2 - y1 + th) // 2
        cv2.putText(img, label, (tx, ty), font, font_scale, self.COLOR_TEXT,
                    thickness, cv2.LINE_AA)
        # Register button rect for hit-testing
        return (x1, y1, x2, y2)

    # ====================================================================
    # UI rendering with buttons
    # ====================================================================
    def render_ui(self, predicted, probabilities, processed_img, is_empty):
        # Layout: left drawing area + right info panel
        drawing_w = int(self.window_width * 0.62)
        drawing_w = min(drawing_w, self.window_height)
        drawing_h = self.window_height

        # Drawing area
        display_canvas = cv2.resize(self.canvas, (drawing_w, drawing_h),
                                    interpolation=cv2.INTER_AREA)
        # Subtle grid
        grid_color = (240, 240, 240)
        for i in range(1, 4):
            y = i * drawing_h // 4
            cv2.line(display_canvas, (0, y), (drawing_w, y), grid_color, 1)

        # Info panel
        panel_w = self.window_width - drawing_w
        panel = np.ones((self.window_height, panel_w, 3), dtype=np.uint8) * 245
        self.buttons = {}

        scale_w = panel_w / 400.0
        font_scale = max(0.7, min(1.4, 0.9 * scale_w))
        thickness = max(2, int(2 * scale_w))

        # ----- Section A: Title -----
        title_y = 40
        if is_empty:
            title = 'No digit'
            color = (0, 0, 200)
        else:
            title = f'Prediction: {predicted}'
            color = (0, 130, 0)
        (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX,
                                       font_scale, thickness)
        cv2.putText(panel, title, ((panel_w - tw) // 2, title_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness,
                    cv2.LINE_AA)

        # ----- Section B: Confidence bars (fixed compact layout) -----
        bar_section_y = 75
        bar_height = 22
        bar_spacing = 28
        margin = max(12, panel_w // 40)
        pct_label_w = 80  # fixed space for percentage labels
        bar_max_w = panel_w - 3 * margin - pct_label_w
        bar_x_start = margin + 20  # after digit label

        for i in range(10):
            prob = probabilities[i].item() * 100
            bar_w = int(prob * bar_max_w / 100.0)
            y = bar_section_y + i * bar_spacing

            # Background bar
            cv2.rectangle(panel, (bar_x_start, y),
                          (bar_x_start + bar_max_w, y + bar_height),
                          (220, 220, 220), -1)
            # Filled bar
            if bar_w > 0:
                bar_color = (60, 180, 60) if i == predicted else (160, 160, 160)
                cv2.rectangle(panel, (bar_x_start, y),
                              (bar_x_start + bar_w, y + bar_height),
                              bar_color, -1)
            # Digit label (0-9)
            cv2.putText(panel, f'{i}',
                        (margin, y + bar_height - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8,
                        (0, 0, 0), 2, cv2.LINE_AA)
            # Percentage label (always fits)
            pct_text = f'{prob:.1f}%'
            cv2.putText(panel, pct_text,
                        (bar_x_start + bar_max_w + 8, y + bar_height - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7,
                        (50, 50, 50), 1, cv2.LINE_AA)

        # ----- Section C: Buttons (Clear + Brush controls) -----
        btn_section_y = self.window_height - 105
        btn_h = 36
        btn_pad = 8

        # Clear button (full width)
        clear_x1 = margin
        clear_x2 = panel_w - margin
        clear_y1 = btn_section_y
        clear_y2 = btn_section_y + btn_h
        state = self.BTN_HOVER if self.hovered_button == 'clear' else self.BTN_NORMAL
        self.buttons['clear'] = self.draw_button(panel, clear_x1, clear_y1,
                                                  clear_x2, clear_y2, 'CLEAR',
                                                  state)

        # Brush controls row
        brush_y1 = btn_section_y + btn_h + btn_pad
        brush_y2 = brush_y1 + btn_h
        brush_btn_w = (panel_w - 3 * margin - 80) // 2  # leave 80px for size label

        # Brush - button
        minus_x1 = margin
        minus_x2 = margin + brush_btn_w
        state = self.BTN_HOVER if self.hovered_button == 'brush_minus' else self.BTN_NORMAL
        self.buttons['brush_minus'] = self.draw_button(
            panel, minus_x1, brush_y1, minus_x2, brush_y2, '-', state
        )

        # Brush size label (centered)
        size_x1 = minus_x2 + 10
        size_x2 = size_x1 + 80
        cv2.rectangle(panel, (size_x1, brush_y1), (size_x2, brush_y2),
                      (255, 255, 255), -1)
        cv2.rectangle(panel, (size_x1, brush_y1), (size_x2, brush_y2),
                      self.COLOR_ACCENT, 1)
        size_text = f'{self.base_brush_radius}'
        (tw, th), _ = cv2.getTextSize(size_text, cv2.FONT_HERSHEY_SIMPLEX,
                                       font_scale * 0.9, 2)
        cv2.putText(panel, size_text,
                    (size_x1 + (size_x2 - size_x1 - tw) // 2,
                     brush_y1 + (brush_y2 - brush_y1 + th) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.9,
                    self.COLOR_TEXT, 2, cv2.LINE_AA)

        # Brush + button
        plus_x1 = size_x2 + 10
        plus_x2 = panel_w - margin
        state = self.BTN_HOVER if self.hovered_button == 'brush_plus' else self.BTN_NORMAL
        self.buttons['brush_plus'] = self.draw_button(
            panel, plus_x1, brush_y1, plus_x2, brush_y2, '+', state
        )

        # Brush size label
        brush_label_y = brush_y2 + 20
        cv2.putText(panel, 'Brush size',
                    (margin, brush_label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.55,
                    (120, 120, 120), 1, cv2.LINE_AA)
        # Visual brush indicator (small filled circle)
        ind_size = 22
        ind_x = panel_w - margin - ind_size // 2
        ind_y = brush_label_y - 5
        cv2.circle(panel, (ind_x, ind_y),
                   max(3, int(self.base_brush_radius * scale_w / 3)),
                   (0, 0, 0), -1, cv2.LINE_AA)

        # ----- Section D: 28x28 preview (safely below bars) -----
        preview_size = max(70, min(110, panel_w // 3))
        preview_y = bar_section_y + 10 * bar_spacing + 15  # just below last bar
        preview_x = (panel_w - preview_size) // 2
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

        return np.hstack([display_canvas, panel])

    # ====================================================================
    # Mouse: handle drawing AND button clicks
    # ====================================================================
    def _hit_test_buttons(self, x, y):
        """Check if (x, y) is inside any button. Return button name or None."""
        for name, (x1, y1, x2, y2) in self.buttons.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                return name
        return None

    def mouse_callback(self, event, x, y, flags, param):
        # Buttons are in panel (right side of window)
        # Convert window coords to panel coords
        drawing_w = int(self.window_width * 0.62)
        drawing_w = min(drawing_w, self.window_height)
        panel_x = x - drawing_w
        panel_y = y

        # Track hover state on any mouse move
        if event == cv2.EVENT_MOUSEMOVE:
            if panel_x >= 0 and self.mode_in_panel(panel_x, panel_y):
                btn = self._hit_test_buttons(panel_x, panel_y)
                if btn != self.hovered_button:
                    self.hovered_button = btn
                    self.need_update = True
            else:
                if self.hovered_button is not None:
                    self.hovered_button = None
                    self.need_update = True
            # Drawing
            if getattr(self, 'drawing', False) and getattr(self, 'last_point', None):
                cx, cy = self.window_to_canvas(x, y)
                self.draw_smooth_segment(self.canvas, self.last_point, (cx, cy),
                                         self._current_brush_radius())
                self.last_point = (cx, cy)
                self.need_update = True

        elif event == cv2.EVENT_LBUTTONDOWN:
            # Check button click
            if panel_x >= 0 and self.mode_in_panel(panel_x, panel_y):
                btn = self._hit_test_buttons(panel_x, panel_y)
                if btn:
                    self._handle_button(btn)
                    return
            # Otherwise, start drawing
            self.drawing = True
            cx, cy = self.window_to_canvas(x, y)
            self.last_point = (cx, cy)
            radius = self._current_brush_radius()
            cv2.circle(self.canvas, (cx, cy), radius, (0, 0, 0), -1, cv2.LINE_AA)
            self.prob_history.clear()
            self.need_update = True
            self.manual_trigger = False

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.last_point = None
            self.need_update = True
            self.manual_trigger = False

    def mode_in_panel(self, panel_x, panel_y):
        """Check if coords are within info panel area (only y matters)."""
        return 0 <= panel_y < self.window_height

    def window_to_canvas(self, x, y):
        drawing_w = int(self.window_width * 0.62)
        drawing_w = min(drawing_w, self.window_height)
        scale = self.canvas_size / drawing_w if drawing_w > 0 else 1.0
        if x >= drawing_w:
            return None, None
        cx = int(x * scale)
        cy = int(y * (self.canvas_size / self.window_height))
        cx = max(0, min(cx, self.canvas_size - 1))
        cy = max(0, min(cy, self.canvas_size - 1))
        return cx, cy

    def _current_brush_radius(self):
        drawing_w = int(self.window_width * 0.62)
        drawing_w = min(drawing_w, self.window_height)
        scale = self.canvas_size / drawing_w if drawing_w > 0 else 1.0
        return max(3, int(self.base_brush_radius / scale))

    def _handle_button(self, name):
        if name == 'clear':
            self.canvas = np.ones((self.canvas_size, self.canvas_size, 3),
                                  dtype=np.uint8) * 255
            self.prob_history.clear()
            self.need_update = True
            self.manual_trigger = False
            print('[INFO] Canvas cleared (button)')
        elif name == 'brush_minus':
            self.base_brush_radius = max(self.min_brush,
                                         self.base_brush_radius - 2)
            self.need_update = True
            print(f'[INFO] Brush size: {self.base_brush_radius}')
        elif name == 'brush_plus':
            self.base_brush_radius = min(self.max_brush,
                                         self.base_brush_radius + 2)
            self.need_update = True
            print(f'[INFO] Brush size: {self.base_brush_radius}')

    # ====================================================================
    # Main loop
    # ====================================================================
    def run(self):
        window_name = 'MNIST Real-time Recognition v2.5 (UI Improved)'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.window_width, self.window_height)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        print('\n' + '=' * 68)
        print(' MNIST Real-time Recognition v2.5 (UI Improved)')
        print('=' * 68)
        print(' Mouse:')
        print('   - Left-click on drawing area (left) to draw')
        print('   - Click CLEAR button to reset canvas')
        print('   - Click [-] / [+] to adjust brush size')
        print(' Keyboard (still works):')
        print('   c, s, +/-, r, ESC')
        print('=' * 68 + '\n')

        last_key_state = {}
        while True:
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
            if self.need_update and (now - self.last_predict_time) >= self.predict_interval:
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
                frame = self.render_ui(predicted, probabilities, processed_img, is_empty)
                cv2.imshow(window_name, frame)
                self.need_update = False
                self.last_predict_time = now

                if self.manual_trigger:
                    print(f'[Manual] pred={predicted}, black_pixels={black_pixels}')
                    for i, p in enumerate(probabilities):
                        print(f'  {i}: {p.item() * 100:5.2f}%')
                    print('-' * 40)
                    self.manual_trigger = False

            key = cv2.waitKey(20) & 0xFF
            if key == 27:
                print('[INFO] ESC pressed.')
                break
            elif key == ord('c'):
                self.canvas = np.ones((self.canvas_size, self.canvas_size, 3),
                                      dtype=np.uint8) * 255
                self.prob_history.clear()
                self.need_update = True
                self.manual_trigger = False
                print('[INFO] Canvas cleared (key).')
            elif key == ord('s'):
                if not last_key_state.get('s', False):
                    self.need_update = True
                    self.manual_trigger = True
                    print('[INFO] Manual trigger.')
                last_key_state['s'] = True
            else:
                last_key_state['s'] = False

            if key == ord('+') or key == ord('='):
                self.base_brush_radius = min(self.max_brush,
                                             self.base_brush_radius + 2)
                self.need_update = True
                print(f'[INFO] Brush size: {self.base_brush_radius}')
            elif key == ord('-') or key == ord('_'):
                self.base_brush_radius = max(self.min_brush,
                                             self.base_brush_radius - 2)
                self.need_update = True
                print(f'[INFO] Brush size: {self.base_brush_radius}')
            elif key == ord('r'):
                self.prob_history.clear()
                self.need_update = True
                print('[INFO] Smoothing reset.')

        cv2.destroyAllWindows()
        print('[INFO] Goodbye.')


if __name__ == '__main__':
    try:
        demo = OpenCVDemoV25(model_path='models/simple_cnn_best.pth',
                              canvas_size=500)
        demo.run()
    except KeyboardInterrupt:
        print('\n[INFO] Interrupted.')
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)




