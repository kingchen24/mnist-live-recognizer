"""Test that OpenCVDemoV2 instantiates correctly (no GUI)."""
import sys
import os
sys.path.insert(0, '.')
from realtime_opencv_v2 import OpenCVDemoV2

print('[1] Creating demo instance...')
demo = OpenCVDemoV2(model_path='models/simple_cnn_best.pth', canvas_size=500)
print(f'    Canvas size: {demo.canvas_size}')
print(f'    Brush radius: {demo.base_brush_radius}')
print(f'    Device: {demo.device}')
print(f'    Model loaded: {type(demo.model).__name__}')
print(f'    Empty threshold: {demo.empty_canvas_threshold}')
print(f'    Predict interval: {demo.predict_interval*1000:.0f} ms')
print(f'    Prob history capacity: {demo.prob_history.maxlen}')

print('\n[2] Testing preprocess with empty canvas...')
demo.canvas[:] = 255
import numpy as np
tensor, processed = demo.preprocess_canvas()
print(f'    Tensor: {tensor}')
print(f'    Processed image sum: {processed.sum()}')

print('\n[3] Testing preprocess with simulated "7" in corner...')
demo.canvas[:] = 255
import cv2
cv2.line(demo.canvas, (60, 80), (160, 80), (0, 0, 0), 15)
cv2.line(demo.canvas, (160, 80), (100, 180), (0, 0, 0), 15)
tensor, processed = demo.preprocess_canvas()
print(f'    Tensor shape: {tensor.shape}, device: {tensor.device}')
print(f'    Processed image shape: {processed.shape}')
print(f'    Tensor min/max/mean: {tensor.min():.3f} / {tensor.max():.3f} / {tensor.mean():.3f}')

print('\n[4] Running inference on the "7"...')
import torch
with torch.no_grad():
    outputs = demo.model(tensor)
    probs = torch.nn.functional.softmax(outputs, dim=1)[0]
    pred = probs.argmax().item()
print(f'    Prediction: {pred} (confidence: {probs[pred].item()*100:.1f}%)')
print(f'    All probabilities:')
for i, p in enumerate(probs):
    bar = chr(0x2588) * int(p.item() * 50)
    print(f'      {i}: {p.item()*100:5.2f}% {bar}')

print('\n[5] Testing probability smoothing...')
demo.prob_history.clear()
for _ in range(5):
    probs_out, pred_out, _ = demo.predict()
print(f'    History size after 5 predictions: {len(demo.prob_history)}')
print(f'    Smoothed prediction: {pred_out}')

print('\n[6] Testing smooth_line drawing...')
demo.canvas[:] = 255
demo.draw_smooth_segment(demo.canvas, (100, 100), (400, 400), 20)
black_pixels = np.sum(demo.canvas[:, :, 0] < 50)
print(f'    Drawn pixels: {black_pixels}')
print(f'    Smooth line OK: {black_pixels > 100}')

print('\n[OK] All v2 components working!')
