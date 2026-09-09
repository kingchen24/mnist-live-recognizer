"""Test v3 components without actual GUI/camera."""
import sys
import os
sys.path.insert(0, '.')
import ast

# Syntax check
with open('realtime_opencv_v3.py') as f:
    ast.parse(f.read())
print('[1/4] Syntax valid')

# Class structure
import importlib.util
spec = importlib.util.spec_from_file_location('v3', 'realtime_opencv_v3.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print('[2/4] Module loads OK')
print(f'      Mode constants: MODE_MOUSE={mod.OpenCVDemoV3.MODE_MOUSE!r}, MODE_CAMERA={mod.OpenCVDemoV3.MODE_CAMERA!r}')

# Instantiate (without opening camera)
demo = mod.OpenCVDemoV3(
    model_path='models/simple_cnn_best.pth',
    canvas_size=500,
    camera_id=0
)
print(f'[3/4] Demo instantiated')
print(f'      mode={demo.mode}, device={demo.device}, roi_size={demo.roi_size}')

# Test camera preprocessing with synthetic frame
import numpy as np
import cv2

print('[4/4] Testing camera preprocessing with synthetic frame...')

# Create a synthetic 640x480 "camera frame" with a digit drawn on paper
# (white background, black "7" in the center ROI area)
synthetic = np.ones((480, 640, 3), dtype=np.uint8) * 240  # paper-ish white
# Draw a "7" in the ROI area (center 240x240 -> top-left at (200, 120))
roi_x, roi_y = 200, 120
roi_size = 240
# horizontal top
cv2.line(synthetic, (roi_x + 40, roi_y + 50),
         (roi_x + 200, roi_y + 50), (30, 30, 30), 14)
# diagonal
cv2.line(synthetic, (roi_x + 200, roi_y + 50),
         (roi_x + 100, roi_y + 200), (30, 30, 30), 14)

# Save for visualization
cv2.imwrite('../demo/synthetic_camera_frame.png', synthetic)

# Run camera preprocessing
demo.roi_size = roi_size
demo.roi_offset = (0, 0)  # center
tensor, processed = demo.preprocess_camera_frame(synthetic)
print(f'      Tensor: {tensor.shape if tensor is not None else None}')
print(f'      Processed (28x28): sum={processed.sum()}')

# Predict
if tensor is not None:
    import torch
    with torch.no_grad():
        outputs = demo.model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        pred = probs.argmax().item()
    print(f'      Prediction: {pred} (confidence: {probs[pred].item()*100:.1f}%)')

# Test rendering UI without showing
print()
print('Test rendering (no display):')
is_empty = tensor is None
probs_cpu = torch.zeros(10) if is_empty else probs.cpu()
pred_int = 0 if is_empty else pred
frame_mouse = demo.render_mouse_ui(
    pred_int, probs_cpu, processed, is_empty
)
frame_cam = demo.render_camera_ui(
    synthetic, pred_int, probs_cpu, processed, is_empty
)
print(f'      Mouse UI frame shape: {frame_mouse.shape}')
print(f'      Camera UI frame shape: {frame_cam.shape}')
print()
print('[OK] All v3 tests pass!')
