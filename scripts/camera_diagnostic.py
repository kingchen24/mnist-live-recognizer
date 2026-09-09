"""Camera diagnostic tool: detect available cameras."""
import cv2
import sys
import time
import os

print('=' * 70)
print(' Camera Diagnostic Tool')
print('=' * 70)
print(f'  OpenCV version: {cv2.__version__}')
print(f'  Python: {sys.version.split()[0]}')
print()

# List available cameras
print('[1] Scanning camera indices 0-4...')
print('-' * 70)
working = []
for i in range(5):
    cap = cv2.VideoCapture(i)
    if not cap.isOpened():
        print(f'  Index {i}: NOT AVAILABLE (cannot open)')
        continue
    # Try to read a frame
    time.sleep(0.3)
    ret, frame = cap.read()
    if not ret or frame is None:
        print(f'  Index {i}: OPENED but no frame (driver issue)')
        cap.release()
        continue
    h, w = frame.shape[:2]
    # Check if frame is actually valid (not all black)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = gray.mean()
    std_brightness = gray.std()
    if mean_brightness < 5 and std_brightness < 5:
        status = f'BLACK FRAME (mean={mean_brightness:.1f}, std={std_brightness:.1f})'
        color_code = '?'
    else:
        status = f'OK (mean={mean_brightness:.1f}, std={std_brightness:.1f})'
        color_code = '+'
        working.append(i)
    print(f'  Index {i}: {color_code} {w}x{h}, {status}')
    cap.release()

print()
print('=' * 70)
if working:
    print(f' [OK] Working cameras: {working}')
    print(f' Use --camera {working[0]} to launch v3 with this camera.')
else:
    print(' [ERROR] No working cameras found!')
    print()
    print(' Troubleshooting steps:')
    print(' 1. Make sure no other app is using the camera')
    print('    (close Zoom, Teams, Skype, Discord, etc.)')
    print(' 2. Windows Settings > Privacy > Camera')
    print('    Make sure "Allow desktop apps to access camera" is ON')
    print(' 3. Open Windows built-in "Camera" app to test the webcam')
    print(' 4. Check if your webcam is properly connected (USB)')
    print(' 5. Update or reinstall webcam driver')
