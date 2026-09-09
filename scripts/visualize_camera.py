"""Visualize camera-mode preprocessing pipeline."""
import sys
import os
sys.path.insert(0, '.')
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load the saved synthetic frame
synthetic = cv2.imread('../demo/synthetic_camera_frame.png')
gray = cv2.cvtColor(synthetic, cv2.COLOR_BGR2GRAY)

# Apply the same pipeline as v3
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
binary = cv2.adaptiveThreshold(
    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV, 11, 2
)

# Find largest contour
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest = max(contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(largest)

# Cropped & padded
pad = int(max(w, h) * 0.15)
cropped = binary[max(0, y-pad):y+h+pad, max(0, x-pad):x+w+pad]
ch, cw = cropped.shape
side = max(ch, cw)
top = (side-ch)//2; bot = side-ch-top
left = (side-cw)//2; right = side-cw-left
square = cv2.copyMakeBorder(cropped, top, bot, left, right, cv2.BORDER_CONSTANT, value=0)
digit_20 = cv2.resize(square, (20, 20), interpolation=cv2.INTER_AREA)
canvas28 = np.zeros((28, 28), dtype=np.uint8)
canvas28[4:24, 4:24] = digit_20

# Visualize 6 stages
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
stages = [
    ('1. Camera Frame', synthetic, 'bgr'),
    ('2. Grayscale', gray, 'gray'),
    ('3. Adaptive Threshold', binary, 'gray'),
    ('4. Largest Contour (bbox)', cv2.cvtColor(cv2.rectangle(synthetic.copy(), (x, y), (x+w, y+h), (0, 0, 255), 3), cv2.COLOR_BGR2RGB), 'rgb'),
    ('5. Cropped + Square-padded', square, 'gray'),
    ('6. Final 28x28 (Model Input)', canvas28, 'gray'),
]
for ax, (title, img, mode) in zip(axes.flat, stages):
    if mode == 'gray':
        ax.imshow(img, cmap='gray')
    elif mode == 'rgb':
        ax.imshow(img)
    else:  # bgr
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(title, fontsize=12)
    ax.axis('off')

plt.suptitle('Camera-Mode Preprocessing Pipeline', fontsize=14)
plt.tight_layout()
plt.savefig('../demo/camera_pipeline.png', dpi=80, bbox_inches='tight')
print('Saved: demo/camera_pipeline.png')
