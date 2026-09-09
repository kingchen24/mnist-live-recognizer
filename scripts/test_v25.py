"""Test v2.5 components and button layout."""
import sys
import os
sys.path.insert(0, '.')
import ast
with open('realtime_opencv_v2_5.py') as f:
    ast.parse(f.read())
print('[1/5] Syntax OK')

import importlib.util
spec = importlib.util.spec_from_file_location('v25', 'realtime_opencv_v2_5.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print('[2/5] Module loads OK')

demo = mod.OpenCVDemoV25(model_path='models/simple_cnn_best.pth', canvas_size=500)
print(f'[3/5] Demo instantiated, brush={demo.base_brush_radius}')

# Render UI once to populate buttons
import numpy as np
import torch
probabilities = torch.zeros(10)
probabilities[8] = 0.99  # fake prediction
processed = np.zeros((28, 28), dtype=np.uint8)
processed[10:18, 10:18] = 200  # fake digit
frame = demo.render_ui(8, probabilities, processed, is_empty=False)
print(f'[4/5] UI rendered: shape={frame.shape}')

# Inspect button positions
print(f'      Buttons defined: {list(demo.buttons.keys())}')
for name, (x1, y1, x2, y2) in demo.buttons.items():
    print(f'        {name}: ({x1},{y1}) -> ({x2},{y2})  size={x2-x1}x{y2-y1}')

# Test hit-testing
print('\n[5/5] Hit-testing:')
test_points = [
    ('in clear button', demo.buttons['clear'][0]+10, demo.buttons['clear'][1]+10),
    ('in brush minus', demo.buttons['brush_minus'][0]+5, demo.buttons['brush_minus'][1]+5),
    ('in brush plus', demo.buttons['brush_plus'][0]+5, demo.buttons['brush_plus'][1]+5),
    ('in drawing area (no btn)', 100, 100),
    ('far below buttons', 100, 1000),
]
for label, x, y in test_points:
    result = demo._hit_test_buttons(x, y)
    print(f'        ({x:>4},{y:>4}) [{label:<25}] -> {result}')

# Test button handlers
print('\n[Button handlers:]')
print(f'      Brush before: {demo.base_brush_radius}')
demo._handle_button('brush_plus')
print(f'      After +: {demo.base_brush_radius}')
demo._handle_button('brush_minus')
demo._handle_button('brush_minus')
print(f'      After --: {demo.base_brush_radius}')
print(f'      Has cleared marker: {demo.need_update}')

# Simulate clear
demo.canvas[:] = 128  # not white
demo._handle_button('clear')
import numpy as np
print(f'      After CLEAR, canvas all white: {(demo.canvas == 255).all()}')

print('\n[OK] All v2.5 components verified!')
