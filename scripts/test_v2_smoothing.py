"""Test EMA smoothing with multiple predictions."""
import sys
import os
sys.path.insert(0, '.')
from realtime_opencv_v2 import OpenCVDemoV2
import numpy as np
import torch

demo = OpenCVDemoV2(model_path='models/simple_cnn_best.pth', canvas_size=500)
demo.prob_history.clear()

# Draw a "3" on canvas
import cv2
demo.canvas[:] = 255
cv2.line(demo.canvas, (100, 100), (200, 100), (0, 0, 0), 18)  # top
cv2.line(demo.canvas, (200, 100), (160, 180), (0, 0, 0), 18)  # upper diagonal
cv2.line(demo.canvas, (160, 180), (200, 260), (0, 0, 0), 18)  # lower diagonal
cv2.line(demo.canvas, (200, 260), (100, 260), (0, 0, 0), 18)  # bottom

print('Simulating 10 predictions on the same "3":')
predictions = []
for i in range(10):
    probs, pred, _ = demo.predict()
    predictions.append(pred)
    top2 = torch.topk(probs, 2)
    if i < 3 or i == 9:
        print(f'  Frame {i+1}: pred={pred}, history={len(demo.prob_history)}, '
              f'top1={top2.indices[0].item()}({top2.values[0].item()*100:.1f}%) '
              f'top2={top2.indices[1].item()}({top2.values[1].item()*100:.1f}%)')

print()
print(f'Predictions over 10 frames: {predictions}')
print(f'All same? {len(set(predictions)) == 1}')
print(f'Final smoothed prediction: {predictions[-1]}')

# Test the empty canvas handling
print()
demo.prob_history.clear()
demo.canvas[:] = 255
probs, pred, processed = demo.predict()
print(f'Empty canvas -> prediction: {pred} (probs[0] = {probs[0].item()*100:.1f}%)')
print('OK')
