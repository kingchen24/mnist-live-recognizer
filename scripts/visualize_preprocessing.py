"""Visualize v1 vs v2 preprocessing side-by-side."""
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
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

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

    # Pick 8 interesting samples
    indices = [3, 5, 7, 12, 21, 27, 35, 42]
    canvas_size = 500

    fig, axes = plt.subplots(4, 8, figsize=(20, 10))

    for col, idx in enumerate(indices):
        img, label = test_dataset[idx]
        digit_28x28 = ((img.squeeze().numpy() * 0.3081 + 0.1307) * 255).astype(np.uint8)
        digit_inverted = 255 - digit_28x28

        # Simulate user drawing in corner of canvas
        canvas = np.ones((canvas_size, canvas_size), dtype=np.uint8) * 255
        scaled = cv2.resize(digit_inverted, (160, 160), interpolation=cv2.INTER_AREA)
        canvas[40:200, 40:200] = scaled
        canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

        # v1: naive resize to 28x28
        gray = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2GRAY)
        v1_img = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
        v1_img = 255 - v1_img

        # v2: improved
        _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        coords = cv2.findNonZero(binary)
        x, y, w, h = cv2.boundingRect(coords)
        pad = int(max(w, h) * 0.15)
        cropped = binary[max(0, y-pad):y+h+pad, max(0, x-pad):x+w+pad]
        ch, cw = cropped.shape
        side = max(ch, cw)
        top, bot = (side-ch)//2, side-ch-(side-ch)//2
        left, right = (side-cw)//2, side-cw-(side-cw)//2
        square = cv2.copyMakeBorder(cropped, top, bot, left, right, cv2.BORDER_CONSTANT, value=0)
        v2_20 = cv2.resize(square, (20, 20), interpolation=cv2.INTER_AREA)
        v2_img = np.zeros((28, 28), dtype=np.uint8)
        v2_img[4:24, 4:24] = v2_20

        # Predictions
        with torch.no_grad():
            t1 = torch.FloatTensor(((v1_img.astype(np.float32)/255 - 0.1307)/0.3081)).unsqueeze(0).unsqueeze(0).to(device)
            t2 = torch.FloatTensor(((v2_img.astype(np.float32)/255 - 0.1307)/0.3081)).unsqueeze(0).unsqueeze(0).to(device)
            p1 = model(t1).argmax(1).item()
            p2 = model(t2).argmax(1).item()
            conf1 = float(torch.softmax(model(t1), dim=1).max().item())
            conf2 = float(torch.softmax(model(t2), dim=1).max().item())

        # Row 0: original (in canvas)
        axes[0, col].imshow(canvas, cmap='gray')
        axes[0, col].set_title(f'True: {label}', fontsize=11)
        axes[0, col].axis('off')

        # Row 1: what v1 produces
        axes[1, col].imshow(v1_img, cmap='gray')
        color1 = 'green' if p1 == label else 'red'
        axes[1, col].set_title(f'v1: pred={p1} ({conf1*100:.0f}%)', fontsize=11, color=color1)
        axes[1, col].axis('off')

        # Row 2: what v2 produces
        axes[2, col].imshow(v2_img, cmap='gray')
        color2 = 'green' if p2 == label else 'red'
        axes[2, col].set_title(f'v2: pred={p2} ({conf2*100:.0f}%)', fontsize=11, color=color2)
        axes[2, col].axis('off')

        # Row 3: v2 with bbox overlay on canvas
        canvas_show = canvas.copy()
        cv2.rectangle(canvas_show, (x-pad, y-pad), (x+w+pad, y+h+pad), (0, 0, 255), 3)
        axes[3, col].imshow(canvas_show, cmap='gray')
        axes[3, col].set_title('v2: bbox detected', fontsize=11, color='blue')
        axes[3, col].axis('off')

    plt.suptitle('Preprocessing Comparison: v1 (naive resize) vs v2 (bbox+center)',
                 fontsize=14, y=0.99)
    plt.tight_layout()
    plt.savefig('../demo/preprocessing_comparison.png', dpi=80, bbox_inches='tight')
    print('Saved: demo/preprocessing_comparison.png')

if __name__ == '__main__':
    main()
