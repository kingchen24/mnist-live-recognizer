# v3.5 placeholder
"""
MNIST Real-time Recognition Demo - v3.5

Key changes from v3:
  - FIX: Camera mode updates continuously (critical bug fix)
  - NEW: Auto-detect available cameras at startup
  - NEW: Press n to cycle cameras at runtime
  - NEW: On-screen buttons (CLEAR, brush +/-, camera selector, freeze)
  - FIX: Confidence percentages no longer cut off

Run: python src/realtime_opencv_v3_5.py [--camera-id N]
"""

import sys
import os
import time
import argparse
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


class OpenCVDemoV35:
    MODE_MOUSE = 'mouse'
    MODE_CAMERA = 'camera'
    BTN_NORMAL = (220, 220, 225)
    BTN_HOVER = (180, 200, 240)
    BTN_ACTIVE = (120, 160, 220)
    BTN_TEXT = (40, 40, 40)
    BTN_BORDER = (60, 60, 60)

    def __init__(self, model_path='models/simple_cnn_best.pth',
                 canvas_size=500, camera_id=0,
                 frame_w=640, frame_h=480):
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
            acc = checkpoint.get('accuracy', 'N/A')
            print(f'[INFO] Model loaded (training accuracy: {acc}%)')
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
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.cap = None
        self.roi_size = 240
        self.roi_offset = (0, 0)
        self.camera_frozen = False
        self.frozen_frame = None
        print()
        print('[INIT] Detecting available cameras...')
        self.available_cameras = self._detect_cameras()
        if self.available_cameras:
            print(f'[INIT] Found {len(self.available_cameras)} camera(s): {self.available_cameras}')
        else:
            print('[INIT] WARNING: No cameras detected!')
        if camera_id in self.available_cameras:
            self.current_camera_idx = camera_id
        elif self.available_cameras:
            self.current_camera_idx = self.available_cameras[0]
        else:
            self.current_camera_idx = 0
        self.mode = self.MODE_MOUSE
        self.window_width = int(canvas_size * 1.6)
        self.window_height = int(canvas_size * 1.2)
        self.need_update = True
        self.prob_history = deque(maxlen=12)  # longer history for stability
        self.smoothed_probs = None
        self.manual_trigger = False
        self.empty_canvas_threshold = 80
        # Hysteresis: don't change prediction unless new top wins by clear margin
        self.last_stable_pred = None
        self.pred_change_threshold = 0.12  # need 12% lead to switch
        self.pred_hold_frames = 3  # or hold current for N frames
        self.last_predict_time = 0
        self.predict_interval = 1.0 / 30.0
        self.camera_frame_count = 0
        self.camera_fps = 0
        self.last_fps_calc = time.time()
        self.buttons = {}
        self.hovered_button = None

    def _detect_cameras(self):
        available = []
        for i in range(5):
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.mean() > 5:
                        available.append(i)
                cap.release()
            except Exception:
                pass
        return available

    def switch_camera(self):
        if len(self.available_cameras) <= 1:
            print('[INFO] Only one camera available.')
            return
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cur_pos = self.available_cameras.index(self.current_camera_idx)
        next_pos = (cur_pos + 1) % len(self.available_cameras)
        self.current_camera_idx = self.available_cameras[next_pos]
        if self.mode == self.MODE_CAMERA:
            self._open_camera()
        self.prob_history.clear()
        self.need_update = True
        self.camera_frozen = False
        print(f'[INFO] Switched to camera {self.current_camera_idx}')

    def _open_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        for backend in [cv2.CAP_DSHOW, cv2.CAP_ANY]:
            try:
                cap = cv2.VideoCapture(self.current_camera_idx, backend)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_w)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_h)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    for _ in range(5):
                        cap.read()
                    self.cap = cap
                    return True
                cap.release()
            except Exception:
                pass
        return False

    def _enter_camera_mode(self):
        print()
        print('[MODE] Switching to camera...')
        if not self._open_camera():
            print(f'[ERROR] Cannot open camera {self.current_camera_idx}')
            print('  Press n to try other cameras, or check Windows privacy settings.')
            return False
        self.mode = self.MODE_CAMERA
        self.prob_history.clear()
        self.need_update = True
        print(f'[MODE] Camera {self.current_camera_idx} active.')
        return True

    def _enter_mouse_mode(self):
        print()
        print('[MODE] Switching to mouse...')
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.mode = self.MODE_MOUSE
        self.prob_history.clear()
        self.need_update = True

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

    def _current_brush_radius(self):
        drawing_w = int(self.window_width * 0.62)
        drawing_w = min(drawing_w, self.window_height)
        scale = self.canvas_size / drawing_w if drawing_w > 0 else 1.0
        return max(3, int(self.base_brush_radius / scale))

    def window_to_canvas(self, x, y):
        drawing_w = int(self.window_width * 0.62)
        drawing_w = min(drawing_w, self.window_height)
        if x >= drawing_w or x < 0:
            return None, None
        scale = self.canvas_size / drawing_w if drawing_w > 0 else 1.0
        cx = int(x * scale)
        cy = int(y * (self.canvas_size / self.window_height))
        cx = max(0, min(cx, self.canvas_size - 1))
        cy = max(0, min(cy, self.canvas_size - 1))
        return cx, cy

    def preprocess_canvas(self):
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        coords = cv2.findNonZero(binary)
        if coords is None:
            return None, np.zeros((28, 28), dtype=np.uint8)
        return self._crop_and_normalize(binary, coords)

    def preprocess_camera_frame(self, frame):
        x0, y0, w, h = self._get_roi()
        roi = frame[y0:y0 + h, x0:x0 + w]
        if roi.size == 0:
            return None, np.zeros((28, 28), dtype=np.uint8)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Stronger blur to reduce sensor noise
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        # Use a larger block size for adaptive threshold (less sensitive to small changes)
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 4
        )
        # === Morphological cleanup: remove noise + fill gaps ===
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # fill small gaps
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)   # remove specks
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, np.zeros((28, 28), dtype=np.uint8)
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < w * h * 0.02:
            return None, np.zeros((28, 28), dtype=np.uint8)
        x, y, cw, ch = cv2.boundingRect(largest)
        contour_pts = np.array([
            [[x, y]], [[x + cw, y]], [[x + cw, y + ch]], [[x, y + ch]]
        ], dtype=np.int32)
        return self._crop_and_normalize(binary, contour_pts)

    def _crop_and_normalize(self, binary, coords):
        x, y, w, h = cv2.boundingRect(coords)
        pad = int(max(w, h) * 0.15)
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(binary.shape[1], x + w + pad), min(binary.shape[0], y + h + pad)
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

    def _get_roi(self):
        cx = self.frame_w // 2 + self.roi_offset[0]
        cy = self.frame_h // 2 + self.roi_offset[1]
        s = self.roi_size
        x = max(0, min(cx - s // 2, self.frame_w - s))
        y = max(0, min(cy - s // 2, self.frame_h - s))
        return x, y, s, s

    def predict(self, tensor, processed_img):
        if tensor is None:
            self.smoothed_probs = torch.zeros(10)
            self.smoothed_probs[0] = 1.0
            self.last_stable_pred = None  # reset on empty
            return self.smoothed_probs, 0, processed_img
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = F.softmax(outputs, dim=1)[0]
        self.prob_history.append(probs)
        if len(self.prob_history) > 0:
            self.smoothed_probs = torch.stack(list(self.prob_history)).mean(dim=0)
        else:
            self.smoothed_probs = probs
        # === HYSTERESIS: only change prediction if new top wins clearly ===
        top2_vals, top2_idx = torch.topk(self.smoothed_probs, 2)
        new_top = int(top2_idx[0].item())
        margin = (top2_vals[0] - top2_vals[1]).item()
        if self.last_stable_pred is None:
            # First prediction - accept anything
            self.last_stable_pred = new_top
        elif new_top != self.last_stable_pred:
            # Top changed - only switch if it wins by clear margin
            if margin >= self.pred_change_threshold:
                self.last_stable_pred = new_top
            # else: keep showing old prediction
        predicted = self.last_stable_pred
        return self.smoothed_probs.cpu(), predicted, processed_img

    def draw_button(self, img, x1, y1, x2, y2, label, state='normal', text_color=None):
        color = {'normal': self.BTN_NORMAL, 'hover': self.BTN_HOVER,
                 'active': self.BTN_ACTIVE}.get(state, self.BTN_NORMAL)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), self.BTN_BORDER, 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, min(1.0, (y2 - y1) / 40.0))
        thickness = 2
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        tx = x1 + (x2 - x1 - tw) // 2
        ty = y1 + (y2 - y1 + th) // 2
        cv2.putText(img, label, (tx, ty), font, font_scale,
                    text_color or self.BTN_TEXT, thickness, cv2.LINE_AA)
        return (x1, y1, x2, y2)

    def _hit_test_buttons(self, panel_x, panel_y):
        for name, (x1, y1, x2, y2) in self.buttons.items():
            if x1 <= panel_x <= x2 and y1 <= panel_y <= y2:
                return name
        return None

    def _handle_button(self, name):
        if name == 'clear':
            self.canvas = np.ones((self.canvas_size, self.canvas_size, 3),
                                  dtype=np.uint8) * 255
            self.prob_history.clear()
            self.need_update = True
            print('[INFO] Canvas cleared (button)')
        elif name == 'brush_minus':
            self.base_brush_radius = max(self.min_brush, self.base_brush_radius - 2)
            self.need_update = True
            print(f'[INFO] Brush: {self.base_brush_radius}')
        elif name == 'brush_plus':
            self.base_brush_radius = min(self.max_brush, self.base_brush_radius + 2)
            self.need_update = True
            print(f'[INFO] Brush: {self.base_brush_radius}')
        elif name == 'cam_prev':
            self._cycle_camera(-1)
        elif name == 'cam_next':
            self._cycle_camera(1)
        elif name == 'cam_freeze':
            self.camera_frozen = not self.camera_frozen
            print(f'[INFO] Camera frozen: {self.camera_frozen}')

    def _cycle_camera(self, direction):
        if len(self.available_cameras) <= 1:
            print('[INFO] Only one camera available.')
            return
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cur_pos = self.available_cameras.index(self.current_camera_idx)
        next_pos = (cur_pos + direction) % len(self.available_cameras)
        self.current_camera_idx = self.available_cameras[next_pos]
        if self.mode == self.MODE_CAMERA:
            self._open_camera()
        self.prob_history.clear()
        self.need_update = True
        print(f'[INFO] Camera {self.current_camera_idx}')

    def mouse_callback(self, event, x, y, flags, param):
        drawing_w = int(self.window_width * 0.62)
        drawing_w = min(drawing_w, self.window_height)
        panel_x = x - drawing_w
        panel_y = y
        if event == cv2.EVENT_MOUSEMOVE:
            if panel_x >= 0:
                btn = self._hit_test_buttons(panel_x, panel_y)
                if btn != self.hovered_button:
                    self.hovered_button = btn
                    self.need_update = True
            else:
                if self.hovered_button is not None:
                    self.hovered_button = None
                    self.need_update = True
            if (self.mode == self.MODE_MOUSE and self.drawing
                    and self.last_point and 0 <= x < drawing_w):
                cx, cy = self.window_to_canvas(x, y)
                if cx is not None:
                    self.draw_smooth_segment(self.canvas, self.last_point,
                                             (cx, cy), self._current_brush_radius())
                    self.last_point = (cx, cy)
                    self.need_update = True
        elif event == cv2.EVENT_LBUTTONDOWN:
            if panel_x >= 0:
                btn = self._hit_test_buttons(panel_x, panel_y)
                if btn:
                    self._handle_button(btn)
                    return
            if self.mode == self.MODE_MOUSE and 0 <= x < drawing_w:
                self.drawing = True
                cx, cy = self.window_to_canvas(x, y)
                if cx is not None:
                    self.last_point = (cx, cy)
                    cv2.circle(self.canvas, (cx, cy),
                               self._current_brush_radius(), (0, 0, 0), -1, cv2.LINE_AA)
                    self.prob_history.clear()
                    self.need_update = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.last_point = None
            if self.mode == self.MODE_MOUSE:
                self.need_update = True

    def render_ui(self, predicted, probabilities, processed_img, is_empty,
                  camera_frame=None):
        drawing_w = int(self.window_width * 0.62)
        drawing_w = min(drawing_w, self.window_height)
        drawing_h = self.window_height
        if self.mode == self.MODE_CAMERA and camera_frame is not None:
            display = self._render_camera_area(camera_frame, drawing_w, drawing_h)
        else:
            display = self._render_mouse_area(drawing_w, drawing_h)
        panel_w = self.window_width - drawing_w
        panel = self._build_info_panel(predicted, probabilities, processed_img,
                                        is_empty, panel_w)
        return np.hstack([display, panel])

    def _render_mouse_area(self, drawing_w, drawing_h):
        display = cv2.resize(self.canvas, (drawing_w, drawing_h),
                             interpolation=cv2.INTER_AREA)
        for i in range(1, 4):
            y = i * drawing_h // 4
            cv2.line(display, (0, y), (drawing_w, y), (240, 240, 240), 1)
        cv2.putText(display, 'MOUSE MODE', (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 130, 0), 2, cv2.LINE_AA)
        cv2.putText(display, 'Draw with left mouse button',
                    (10, drawing_h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
        return display

    def _render_camera_area(self, frame, drawing_w, drawing_h):
        scale = min(drawing_w / self.frame_w, drawing_h / self.frame_h)
        new_w = int(self.frame_w * scale)
        new_h = int(self.frame_h * scale)
        display = cv2.resize(frame, (new_w, new_h))
        padded = np.ones((drawing_h, drawing_w, 3), dtype=np.uint8) * 40
        x_off = (drawing_w - new_w) // 2
        y_off = (drawing_h - new_h) // 2
        padded[y_off:y_off + new_h, x_off:x_off + new_w] = display
        rx, ry, rw, rh = self._get_roi()
        rx_d = int(rx * scale) + x_off
        ry_d = int(ry * scale) + y_off
        rw_d = int(rw * scale)
        rh_d = int(rh * scale)
        roi_color = (0, 255, 255) if self.camera_frozen else (0, 255, 0)
        cv2.rectangle(padded, (rx_d, ry_d), (rx_d + rw_d, ry_d + rh_d),
                      roi_color, 2)
        corner_len = 18
        for cx, cy, dx, dy in [
            (rx_d, ry_d, 1, 1), (rx_d + rw_d, ry_d, -1, 1),
            (rx_d, ry_d + rh_d, 1, -1), (rx_d + rw_d, ry_d + rh_d, -1, -1)
        ]:
            cv2.line(padded, (cx, cy), (cx + dx * corner_len, cy), roi_color, 3)
            cv2.line(padded, (cx, cy), (cx, cy + dy * corner_len), roi_color, 3)
        mode_text = 'CAMERA MODE (FROZEN)' if self.camera_frozen else 'CAMERA MODE'
        cv2.putText(padded, mode_text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, roi_color, 2, cv2.LINE_AA)
        cv2.putText(padded, f'Camera {self.current_camera_idx} | FPS: {self.camera_fps:.0f}',
                    (10, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(padded, 'Position digit inside green box',
                    (10, drawing_h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        return padded

    def _build_info_panel(self, predicted, probabilities, processed_img,
                          is_empty, width):
        panel = np.ones((self.window_height, width, 3), dtype=np.uint8) * 245
        self.buttons = {}
        margin = max(12, width // 30)
        scale_w = width / 400.0
        font_scale = max(0.7, min(1.4, 0.9 * scale_w))
        thickness = max(2, int(2 * scale_w))
        # Title
        title_y = 40
        if is_empty:
            title = 'No digit'
            color = (0, 0, 200)
        else:
            title = f'Prediction: {predicted}'
            color = (0, 130, 0)
        (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX,
                                       font_scale, thickness)
        cv2.putText(panel, title, ((width - tw) // 2, title_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness,
                    cv2.LINE_AA)
        # Bars
        bar_section_y = 75
        bar_height = 22
        bar_spacing = 28
        pct_label_w = 80
        bar_max_w = width - 3 * margin - pct_label_w
        bar_x_start = margin + 20
        for i in range(10):
            prob = probabilities[i].item() * 100
            bar_w = int(prob * bar_max_w / 100.0)
            y = bar_section_y + i * bar_spacing
            cv2.rectangle(panel, (bar_x_start, y),
                          (bar_x_start + bar_max_w, y + bar_height),
                          (220, 220, 220), -1)
            if bar_w > 0:
                bar_color = (60, 180, 60) if i == predicted else (160, 160, 160)
                cv2.rectangle(panel, (bar_x_start, y),
                              (bar_x_start + bar_w, y + bar_height),
                              bar_color, -1)
            cv2.putText(panel, f'{i}',
                        (margin, y + bar_height - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8,
                        (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(panel, f'{prob:.1f}%',
                        (bar_x_start + bar_max_w + 8, y + bar_height - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7,
                        (50, 50, 50), 1, cv2.LINE_AA)
        # Preview
        preview_size = max(70, min(110, width // 3))
        preview_y = bar_section_y + 10 * bar_spacing + 15
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
        # Mode-specific controls
        if self.mode == self.MODE_MOUSE:
            self._render_mouse_controls(panel, width, margin, font_scale)
        else:
            self._render_camera_controls(panel, width, margin, font_scale)
        return panel

    def _render_mouse_controls(self, panel, width, margin, font_scale):
        btn_h = 34
        btn_pad = 8
        clear_y = self.window_height - 95
        clear_x1, clear_x2 = margin, width - margin
        st = 'hover' if self.hovered_button == 'clear' else 'normal'
        self.buttons['clear'] = self.draw_button(panel, clear_x1, clear_y,
                                                  clear_x2, clear_y + btn_h,
                                                  'CLEAR', st)
        brush_y = clear_y + btn_h + btn_pad
        brush_btn_w = (width - 3 * margin - 70) // 2
        minus_x1, minus_x2 = margin, margin + brush_btn_w
        st = 'hover' if self.hovered_button == 'brush_minus' else 'normal'
        self.buttons['brush_minus'] = self.draw_button(panel, minus_x1, brush_y,
                                                        minus_x2, brush_y + btn_h,
                                                        '-', st)
        size_x1, size_x2 = minus_x2 + 8, minus_x2 + 78
        cv2.rectangle(panel, (size_x1, brush_y), (size_x2, brush_y + btn_h),
                      (255, 255, 255), -1)
        cv2.rectangle(panel, (size_x1, brush_y), (size_x2, brush_y + btn_h),
                      self.BTN_BORDER, 1)
        size_text = f'{self.base_brush_radius}'
        (tw, th), _ = cv2.getTextSize(size_text, cv2.FONT_HERSHEY_SIMPLEX,
                                       font_scale * 0.9, 2)
        cv2.putText(panel, size_text,
                    (size_x1 + (size_x2 - size_x1 - tw) // 2,
                     brush_y + (btn_h + th) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.9,
                    self.BTN_TEXT, 2, cv2.LINE_AA)
        plus_x1, plus_x2 = size_x2 + 8, width - margin
        st = 'hover' if self.hovered_button == 'brush_plus' else 'normal'
        self.buttons['brush_plus'] = self.draw_button(panel, plus_x1, brush_y,
                                                       plus_x2, brush_y + btn_h,
                                                       '+', st)
        cv2.putText(panel, "Press 'v' for camera mode",
                    (margin, brush_y + btn_h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.55,
                    (120, 120, 120), 1, cv2.LINE_AA)

    def _render_camera_controls(self, panel, width, margin, font_scale):
        btn_h = 34
        btn_pad = 8
        title_y = self.window_height - 130
        cam_text = f'Camera {self.current_camera_idx} of {self.available_cameras}'
        cv2.putText(panel, cam_text, (margin, title_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7,
                    self.BTN_TEXT, 1, cv2.LINE_AA)
        btn_y = title_y + 8
        avail = width - 2 * margin
        cam_btn_w = avail // 4
        st = 'hover' if self.hovered_button == 'cam_prev' else 'normal'
        self.buttons['cam_prev'] = self.draw_button(panel, margin, btn_y,
                                                     margin + cam_btn_w,
                                                     btn_y + btn_h, '<', st)
        cam_x1, cam_x2 = margin + cam_btn_w, margin + 2 * cam_btn_w
        cv2.rectangle(panel, (cam_x1, btn_y), (cam_x2, btn_y + btn_h),
                      (245, 245, 250), -1)
        cv2.rectangle(panel, (cam_x1, btn_y), (cam_x2, btn_y + btn_h),
                      self.BTN_BORDER, 1)
        cv2.putText(panel, f'Cam {self.current_camera_idx}',
                    (cam_x1 + 8, btn_y + (btn_h + 10) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8,
                    self.BTN_TEXT, 2, cv2.LINE_AA)
        next_x1, next_x2 = cam_x2, cam_x2 + cam_btn_w
        st = 'hover' if self.hovered_button == 'cam_next' else 'normal'
        self.buttons['cam_next'] = self.draw_button(panel, next_x1, btn_y,
                                                     next_x2, btn_y + btn_h,
                                                     '>', st)
        freeze_x1, freeze_x2 = next_x2, next_x2 + cam_btn_w
        st = 'active' if self.camera_frozen else (
            'hover' if self.hovered_button == 'cam_freeze' else 'normal')
        freeze_label = 'FRZ' if not self.camera_frozen else 'GO'
        self.buttons['cam_freeze'] = self.draw_button(
            panel, freeze_x1, btn_y, freeze_x2, btn_y + btn_h,
            freeze_label, st
        )
        cv2.putText(panel, "Press 'v' for mouse | n=next cam | r=reset ROI",
                    (margin, btn_y + btn_h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.55,
                    (120, 120, 120), 1, cv2.LINE_AA)

    def run(self):
        window_name = 'MNIST Real-time Recognition v3.5 (Mouse + Camera)'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.window_width, self.window_height)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        print()
        print('=' * 68)
        print(' MNIST Real-time Recognition v3.5')
        print('=' * 68)
        print(f' Available cameras: {self.available_cameras}')
        print(f' Starting in: {self.mode} mode')
        print()
        print(' Controls:')
        print('   v               - toggle mouse/camera mode')
        print('   n               - cycle to next camera (camera mode)')
        print('   f               - freeze/unfreeze camera (camera mode)')
        print('   [ / ]           - ROI smaller/larger (camera mode)')
        print('   Arrow keys      - move ROI (camera mode)')
        print('   Mouse: draw / click buttons (CLEAR, brush +/-, cam < >, FREEZE)')
        print('   c, s, +/-, r, ESC (keyboard shortcuts)')
        print('=' * 68)
        print()
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
            throttled = (now - self.last_predict_time) >= self.predict_interval
            # === FIX: camera always processes, mouse only on change ===
            if self.mode == self.MODE_CAMERA and throttled:
                self._process_camera_frame()
                self.last_predict_time = now
            elif self.mode == self.MODE_MOUSE and self.need_update and throttled:
                self._process_mouse_frame()
                self.need_update = False
                self.last_predict_time = now
            key = cv2.waitKey(20) & 0xFF
            if key == 27:
                print('[INFO] ESC pressed, exiting.')
                break
            elif key == ord('v'):
                if self.mode == self.MODE_MOUSE:
                    self._enter_camera_mode()
                else:
                    self._enter_mouse_mode()
            elif key == ord('n') and self.mode == self.MODE_CAMERA:
                self.switch_camera()
            elif key == ord('f') and self.mode == self.MODE_CAMERA:
                self.camera_frozen = not self.camera_frozen
                print(f'[INFO] Camera frozen: {self.camera_frozen}')
            elif key == ord('c') and self.mode == self.MODE_MOUSE:
                self.canvas = np.ones((self.canvas_size, self.canvas_size, 3),
                                      dtype=np.uint8) * 255
                self.prob_history.clear()
                self.need_update = True
            elif key == ord('s'):
                if not last_key_state.get('s', False):
                    self.need_update = True
                    self.manual_trigger = True
                last_key_state['s'] = True
            else:
                last_key_state['s'] = False
            if key == ord('+') or key == ord('='):
                if self.mode == self.MODE_MOUSE:
                    self.base_brush_radius = min(self.max_brush, self.base_brush_radius + 2)
                    self.need_update = True
            elif key == ord('-') or key == ord('_'):
                if self.mode == self.MODE_MOUSE:
                    self.base_brush_radius = max(self.min_brush, self.base_brush_radius - 2)
                    self.need_update = True
            elif key == ord(']') or key == ord('}'):
                if self.mode == self.MODE_CAMERA:
                    self.roi_size = min(480, self.roi_size + 20)
                    self.need_update = True
            elif key == ord('[') or key == ord('{'):
                if self.mode == self.MODE_CAMERA:
                    self.roi_size = max(80, self.roi_size - 20)
                    self.need_update = True
            elif key == 82:
                if self.mode == self.MODE_CAMERA:
                    self.roi_offset = (self.roi_offset[0], self.roi_offset[1] - 20)
                    self.need_update = True
            elif key == 84:
                if self.mode == self.MODE_CAMERA:
                    self.roi_offset = (self.roi_offset[0], self.roi_offset[1] + 20)
                    self.need_update = True
            elif key == 81:
                if self.mode == self.MODE_CAMERA:
                    self.roi_offset = (self.roi_offset[0] - 20, self.roi_offset[1])
                    self.need_update = True
            elif key == 83:
                if self.mode == self.MODE_CAMERA:
                    self.roi_offset = (self.roi_offset[0] + 20, self.roi_offset[1])
                    self.need_update = True
            elif key == ord('r'):
                self.prob_history.clear()
                if self.mode == self.MODE_CAMERA:
                    self.roi_offset = (0, 0)
                self.need_update = True
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print('[INFO] Goodbye.')

    def _process_mouse_frame(self):
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
            probabilities, predicted, processed_img = self.predict(tensor, processed_img)
        frame = self.render_ui(predicted, probabilities, processed_img, is_empty)
        cv2.imshow('MNIST Real-time Recognition v3.5 (Mouse + Camera)', frame)
        if self.manual_trigger:
            print(f'[Manual] pred={predicted}, black={black_pixels}')
            for i, p in enumerate(probabilities):
                print(f'  {i}: {p.item() * 100:5.2f}%')
            print('-' * 40)
            self.manual_trigger = False

    def _process_camera_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return
        if self.camera_frozen and self.frozen_frame is not None:
            frame = self.frozen_frame
        else:
            ret, frame = self.cap.read()
            if not ret:
                return
            if self.camera_frozen:
                self.frozen_frame = frame.copy()
        self.camera_frame_count += 1
        if time.time() - self.last_fps_calc > 1.0:
            self.camera_fps = self.camera_frame_count / (time.time() - self.last_fps_calc)
            self.camera_frame_count = 0
            self.last_fps_calc = time.time()
        tensor, processed_img = self.preprocess_camera_frame(frame)
        is_empty = tensor is None
        if is_empty:
            self.smoothed_probs = torch.zeros(10)
            self.smoothed_probs[0] = 1.0
            self.prob_history.clear()
            probabilities = self.smoothed_probs
            predicted = 0
        else:
            probabilities, predicted, processed_img = self.predict(tensor, processed_img)
        frame_ui = self.render_ui(predicted, probabilities, processed_img, is_empty,
                                   camera_frame=frame)
        cv2.imshow('MNIST Real-time Recognition v3.5 (Mouse + Camera)', frame_ui)
        if self.manual_trigger:
            print(f'[Manual] pred={predicted}, empty={is_empty}')
            for i, p in enumerate(probabilities):
                print(f'  {i}: {p.item() * 100:5.2f}%')
            print('-' * 40)
            self.manual_trigger = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MNIST Real-time Recognition v3.5')
    parser.add_argument('--camera-id', type=int, default=0)
    parser.add_argument('--frame-w', type=int, default=640)
    parser.add_argument('--frame-h', type=int, default=480)
    args = parser.parse_args()
    try:
        demo = OpenCVDemoV35(
            model_path='models/simple_cnn_best.pth',
            canvas_size=500,
            camera_id=args.camera_id,
            frame_w=args.frame_w,
            frame_h=args.frame_h
        )
        demo.run()
    except KeyboardInterrupt:
        print()
        print('[INFO] Interrupted.')
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)



