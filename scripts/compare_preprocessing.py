"""Compare v1 (naive) vs v2 (improved) preprocessing on real MNIST test samples."""
import torch
import sys
import os
sys.path.insert(0, '.')
from model import create_model
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def main():
    import cv2
    import numpy as np

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model('simple_cnn', num_classes=10, device=device)
    checkpoint = torch.load('../models/simple_cnn_best.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    def preprocess_v1(canvas_bgr):
        """Original v1 approach: just resize canvas to 28x28."""
        gray = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
        inverted = 255 - resized
        normalized = inverted.astype(np.float32) / 255.0
        normalized = (normalized - 0.1307) / 0.3081
        return torch.FloatTensor(normalized).unsqueeze(0).unsqueeze(0).to(device)

    def preprocess_v2(canvas_bgr):
        """Improved: bbox crop + aspect ratio + center."""
        gray = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        coords = cv2.findNonZero(binary)
        if coords is None:
            return None
        x, y, w, h = cv2.boundingRect(coords)
        pad = int(max(w, h) * 0.15)
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(canvas_bgr.shape[1], x + w + pad), min(canvas_bgr.shape[0], y + h + pad)
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
        return torch.FloatTensor(normalized).unsqueeze(0).unsqueeze(0).to(device)

    canvas_size = 500
    n_samples = 300

    # Scenarios: digit placed in different positions/sizes on canvas
    scenarios = {
        'corner-140px': (30, 30, 140),       # corner, medium size
        'corner-200px': (30, 30, 200),       # corner, large
        'center-140px': (180, 180, 140),     # center, medium
        'center-200px': (150, 150, 200),     # center, large
        'mid-left-100px': (200, 50, 100),    # mid-left, small
    }

    results = {name: {'v1': 0, 'v2': 0} for name in scenarios}

    for i, (img, label) in enumerate(test_loader):
        if i >= n_samples:
            break
        true_label = label.item()

        # MNIST is white-digit on black-background
        # Simulate user drawing: invert (black digit on white background)
        # then scale to mimic user stroke
        digit_28x28 = img.squeeze().numpy()
        digit_28x28_uint8 = ((digit_28x28 * 0.3081 + 0.1307) * 255).astype(np.uint8)
        digit_inverted = 255 - digit_28x28_uint8  # now looks like user-drawn

        for name, (px, py, size) in scenarios.items():
            canvas = np.ones((canvas_size, canvas_size), dtype=np.uint8) * 255
            scaled = cv2.resize(digit_inverted, (size, size), interpolation=cv2.INTER_AREA)
            canvas[py:py+size, px:px+size] = scaled
            canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

            with torch.no_grad():
                t1 = preprocess_v1(canvas_bgr)
                t2 = preprocess_v2(canvas_bgr)
                p1 = model(t1).argmax(1).item()
                p2 = model(t2).argmax(1).item()

            if p1 == true_label:
                results[name]['v1'] += 1
            if p2 == true_label:
                results[name]['v2'] += 1

    print('=' * 70)
    print(f' Preprocessing comparison on {n_samples} MNIST test samples')
    print('=' * 70)
    print(f'{"Scenario":<22}{"v1 (naive)":>15}{"v2 (improved)":>18}{"Improvement":>15}')
    print('-' * 70)
    for name in scenarios:
        v1 = 100 * results[name]['v1'] / n_samples
        v2 = 100 * results[name]['v2'] / n_samples
        diff = v2 - v1
        sign = '+' if diff >= 0 else ''
        print(f'{name:<22}{v1:>13.2f}%{v2:>16.2f}%{sign}{diff:>13.2f}%')
    print('=' * 70)

if __name__ == '__main__':
    main()
