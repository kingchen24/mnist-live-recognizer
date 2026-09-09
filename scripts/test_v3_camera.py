"""Test v3 camera mode components."""
import sys, os
sys.path.insert(0, '.')
import ast
with open('realtime_opencv_v3.py') as f:
    ast.parse(f.read())
print('[1/5] v3 syntax OK')

import importlib.util
spec = importlib.util.spec_from_file_location('v3', 'realtime_opencv_v3.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print('[2/5] v3 module loads OK')

# Mock cv2 GUI calls
import cv2
import numpy as np
cv2.namedWindow = lambda *a, **k: None
cv2.resizeWindow = lambda *a, **k: None
cv2.setMouseCallback = lambda *a, **k: None
cv2.imshow = lambda *a, **k: None
cv2.waitKey = lambda *a, **k: 255
cv2.destroyAllWindows = lambda: None
cv2.getWindowImageRect = lambda *a, **k: (0, 0, 1280, 500)

demo = mod.OpenCVDemoV3(model_path='models/simple_cnn_best.pth', canvas_size=500)
print(f'[3/5] Demo initialized in {demo.mode} mode')
print(f'      Device: {demo.device}')
print(f'      ROI size: {demo.roi_size}, offset: {demo.roi_offset}')

# Test mode switching (without actually opening camera)
print('\n[4/5] Testing mode labels:')
print(f'      Current mode: {demo.mode}')
print(f'      Mode constants: MOUSE={demo.MODE_MOUSE!r}, CAMERA={demo.MODE_CAMERA!r}')

# Test ROI calculation
x, y, w, h = demo._get_roi()
print(f'      Default ROI (centered): x={x}, y={y}, size={w}x{h}')

# Adjust ROI
demo.roi_size = 300
demo.roi_offset = (50, -30)
x, y, w, h = demo._get_roi()
print(f'      Adjusted ROI: x={x}, y={y}, size={w}x{h}')

# Test synthetic frame
synthetic = np.ones((480, 640, 3), dtype=np.uint8) * 240
# Draw a "3" in ROI
roi_x, roi_y, roi_w, roi_h = demo._get_roi()
demo.roi_size = 240
demo.roi_offset = (0, 0)
roi_x, roi_y, roi_w, roi_h = demo._get_roi()
cv2.line(synthetic, (roi_x + 30, roi_y + 40), (roi_x + 200, roi_y + 40), (30, 30, 30), 14)
cv2.line(synthetic, (roi_x + 200, roi_y + 40), (roi_x + 160, roi_y + 120), (30, 30, 30), 14)
cv2.line(synthetic, (roi_x + 160, roi_y + 120), (roi_x + 200, roi_y + 200), (30, 30, 30), 14)
cv2.line(synthetic, (roi_x + 200, roi_y + 200), (roi_x + 30, roi_y + 200), (30, 30, 30), 14)

tensor, processed = demo.preprocess_camera_frame(synthetic)
print(f'\n[5/5] Synthetic camera frame test:')
print(f'      Tensor: {tensor.shape if tensor is not None else None}')
print(f'      Processed sum: {processed.sum()}')

if tensor is not None:
    import torch
    with torch.no_grad():
        outputs = demo.model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        pred = probs.argmax().item()
    print(f'      Prediction: {pred} (confidence: {probs[pred].item()*100:.1f}%)')

# Test camera UI rendering
demo.mode = demo.MODE_CAMERA
probs_cpu = torch.zeros(10)
if tensor is not None:
    probs_cpu = probs.cpu()
frame = demo.render_camera_ui(
    synthetic, pred, probs_cpu, processed, is_empty=(tensor is None)
)
cv2.imwrite('../demo/v3_camera_test.png', frame)
print('Saved: demo/v3_camera_test.png')
print('\n[OK] All v3 components verified!')
