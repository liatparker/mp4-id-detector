#!/usr/bin/env python3
"""
Download a pre-trained gun detection YOLO model from Roboflow
"""
import os
from ultralytics import YOLO

print("🔫 Downloading Gun Detection Model...")
print("Source: Roboflow Universe - Weapon Detection")
print()

# Option 1: Try to download from Roboflow API (requires API key)
# For now, we'll use a public model or guide user to download manually

print("📋 TO GET A GUN DETECTION MODEL:")
print()
print("1. Visit: https://universe.roboflow.com/weapon-detection")
print("2. Search for 'gun detection' or 'weapon detection'")
print("3. Choose a model (recommended: 'Gun Detection' with 4 classes)")
print("4. Download as YOLOv8 format")
print("5. Save as 'weapon_yolov8.pt' in this directory")
print()
print("Alternative: Use a direct download link if available")
print()

# Check if model exists
if os.path.exists('weapon_yolov8.pt'):
    print("✅ weapon_yolov8.pt found!")
    model = YOLO('weapon_yolov8.pt')
    print(f"   Model loaded successfully")
    print(f"   Classes: {model.names}")
else:
    print("⚠️  weapon_yolov8.pt not found")
    print("   Place your downloaded model in this directory")
    
