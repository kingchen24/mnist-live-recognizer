"""Quick camera test - shows camera feed in a window for 5 seconds."""
import cv2
import sys
import time

print('Testing cameras...')
print('Press Q or ESC to quit')

# Try each camera and pick the first one that works
for cam_id in range(3):
    print(f'\nTrying camera {cam_id}...')
    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)  # CAP_DSHOW is more reliable on Windows
    if not cap.isOpened():
        print(f'  Cannot open camera {cam_id}')
        continue
    print(f'  Opened camera {cam_id}')
    # Read 3 warm-up frames
    for _ in range(3):
        cap.read()
    ret, frame = cap.read()
    if not ret:
        print(f'  Cannot read frame')
        cap.release()
        continue
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(f'  Frame: {w}x{h}, mean brightness: {gray.mean():.1f}')
    if gray.mean() < 5:
        print(f'  Frame is black, trying next...')
        cap.release()
        continue
    # Show live feed for 5 seconds
    print(f'  Showing live feed for 5 seconds (close window or press Q to quit)...')
    print(f'  >>> Your working camera is index {cam_id} <<<')
    start = time.time()
    while time.time() - start < 10:
        ret, frame = cap.read()
        if not ret:
            print('  Lost frame!')
            break
        cv2.putText(frame, f'Camera {cam_id} - press Q to quit',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow(f'Camera Test {cam_id}', frame)
        if cv2.waitKey(30) & 0xFF in [ord('q'), 27]:
            break
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)

print('\nNo working camera found!')
print('Possible issues:')
print('  1. Camera is being used by another app (Zoom, Teams, etc.)')
print('  2. Camera driver not installed')
print('  3. Windows privacy settings blocking access')
