#!/usr/bin/env python
"""Test script to verify all packages are installed and working."""

import sys

print("=" * 50)
print("Testing Soccer Analytics Backend Environment")
print("=" * 50)
print(f"Python Version: {sys.version}")
print()

# Test ML packages
try:
    import torch
    print(f"[OK] PyTorch: {torch.__version__}")
except ImportError as e:
    print(f"[FAIL] PyTorch: Not installed - {e}")

try:
    import torchvision
    print(f"[OK] TorchVision: {torchvision.__version__}")
except ImportError as e:
    print(f"[FAIL] TorchVision: Not installed - {e}")

try:
    import ultralytics
    print(f"[OK] Ultralytics: {ultralytics.__version__}")
except ImportError as e:
    print(f"[FAIL] Ultralytics: Not installed - {e}")

try:
    import cv2
    print(f"[OK] OpenCV: {cv2.__version__}")
except ImportError as e:
    print(f"[FAIL] OpenCV: Not installed - {e}")

print()

# Test Web packages
try:
    import fastapi
    print(f"[OK] FastAPI: {fastapi.__version__}")
except ImportError as e:
    print(f"[FAIL] FastAPI: Not installed - {e}")

try:
    import uvicorn
    print(f"[OK] Uvicorn: {uvicorn.__version__}")
except ImportError as e:
    print(f"[FAIL] Uvicorn: Not installed - {e}")

print()

# Test Data packages
try:
    import pandas as pd
    print(f"[OK] Pandas: {pd.__version__}")
except ImportError as e:
    print(f"[FAIL] Pandas: Not installed - {e}")

try:
    import numpy as np
    print(f"[OK] NumPy: {np.__version__}")
except ImportError as e:
    print(f"[FAIL] NumPy: Not installed - {e}")

print()
print("=" * 50)
print("Environment test complete!")
print("=" * 50)

# Test YOLO model loading
try:
    from ultralytics import YOLO
    import os
    
    models_exist = []
    model_paths = {
        "player": "models/player.pt",
        "ball": "models/ball.pt",
        "field": "models/field.pt"
    }
    
    print("\nChecking model files:")
    for name, path in model_paths.items():
        if os.path.exists(path):
            print(f"[OK] {name} model found: {path}")
            models_exist.append(name)
        else:
            print(f"[WARN] {name} model not found: {path}")
    
    if len(models_exist) == 3:
        print("\n[OK] All models found! Ready to run predictions.")
    else:
        print(f"\n[WARN] Only {len(models_exist)}/3 models found.")
        
except Exception as e:
    print(f"\n[WARN] Could not check models: {e}")

