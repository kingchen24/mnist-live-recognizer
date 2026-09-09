"""Pre-test: simulate mouse-style drawings and check accuracy."""
import sys, os
sys.path.insert(0, '.')
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import create_model
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def main():
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

    def preprocess_v2(canvas_bgr, canvas_size=500):
        gray = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        coords = cv2.findNonZero(binary)
        if coords is None:
            return None
        x, y, w, h = cv2.boundingRect(coords)
        pad = int(max(w, h) * 0.15)
        x1 = max(0, x - pad); y1 = max(0, y - pad)
        x2 = min(canvas_size, x + w + pad); y2 = min(canvas_size, y + h + pad)
        cropped = binary[y1:y2, x1:x2]
        ch, cw = cropped.shape
        side = max(ch, cw)
        top = (side - ch) // 2; bot = side - ch - top
        left = (side - cw) // 2; right = side - cw - left
        square = cv2.copyMakeBorder(cropped, top, bot, left, right, cv2.BORDER_CONSTANT, value=0)
        digit = cv2.resize(square, (20, 20), interpolation=cv2.INTER_AREA)
        canvas28 = np.zeros((28, 28), dtype=np.uint8)
        canvas28[4:24, 4:24] = digit
        norm = canvas28.astype(np.float32) / 255.0
        norm = (norm - 0.1307) / 0.3081
        return torch.FloatTensor(norm).unsqueeze(0).unsqueeze(0).to(device)

    def simulate_drawing(digit_28x28, position, size, simulate_thin=True):
        digit = 255 - ((digit_28x28 * 0.3081 + 0.1307) * 255).astype(np.uint8)
        scaled = cv2.resize(digit, (size, size), interpolation=cv2.INTER_AREA)
        if simulate_thin:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            scaled = cv2.erode(scaled, kernel, iterations=1)
        canvas = np.ones((500, 500), dtype=np.uint8) * 255
        px, py = position
        canvas[py:py+size, px:px+size] = scaled
        return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    print('=' * 70)
    print(' Pre-test: mouse-style simulation (250 samples per scenario)')
    print('=' * 70)
    print(f'{"Scenario":<40}{"Accuracy":>15}')
    print('-' * 70)

    scenarios = [
        ('center, 150px, thin strokes', (175, 175), 150),
        ('center, 200px, thin strokes', (150, 150), 200),
        ('center, 250px, thin strokes', (125, 125), 250),
        ('corner (50,50), 150px, thin', (50, 50), 150),
        ('corner (50,50), 200px, thin', (50, 50), 200),
        ('center, 200px, thick strokes', (150, 150), 200),
    ]

    n_samples = 250
    results = []

    for name, pos, size in scenarios:
        correct = 0
        thin = 'thin' in name
        for i in range(n_samples):
            img, label = test_dataset[i]
            canvas = simulate_drawing(img.squeeze().numpy(), pos, size, thin)
            tensor = preprocess_v2(canvas)
            if tensor is None:
                continue
            with torch.no_grad():
                pred = int(model(tensor).argmax(1).item())
            if pred == int(label):
                correct += 1
        acc = 100 * correct / n_samples
        results.append((name, acc))
        print(f'{name:<40}{acc:>14.2f}%')

    print('-' * 70)
    avg_acc = np.mean([a for _, a in results])
    print(f'{"Average across scenarios":<40}{avg_acc:>14.2f}%')
    print('=' * 70)
    print()
    print('Conclusion: With mouse-style thin strokes, expect ~95-99% accuracy.')
    print('If your strokes are too thin/light, press + to make brush thicker.')

if __name__ == '__main__':
    main()
