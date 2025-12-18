"""\
Training script for YOLOv11 object detection models.

Requirements:
- Install ../requirements.txt in a python virtual environment.

Usage: python3 train.py

Authors: Pedro Pinto, João Pinto, Fedor Chikhachev
"""
import os
from ultralytics import YOLO
from ultralytics import settings
import albumentations as A
import cv2

settings.update({"tensorboard": True})

# -- Training configurations
trainings = [
    {
        "name": "run_yolo11n_768",
        "epochs": 100,
        "imgsz": 768,
        "yolo_version": "yolo11n",
        "batch": -1,    # -1 for auto batch size
    }
]

# -- Training loop
for config in trainings:
    epochs = config["epochs"]
    name = config["name"]
    batch = config["batch"]
    imgsz = config["imgsz"]
    yolo_version = config["yolo_version"]

    # Using P2-head
    # model = YOLO("yolo11n-p2.yaml").load(f'{yolo_version}.pt')
    
    # Without P2-head
    model = YOLO(f"{yolo_version}.pt")

    results = model.train(
        data="dataset.yaml",
        epochs=epochs, 
        imgsz=imgsz,
        batch=batch,
        project="models",
        name=name,

        augmentations=[
            # Blur variants
            A.OneOf([
                A.MotionBlur(blur_limit=7, p=1.0),
                A.MedianBlur(blur_limit=7, p=1.0),
                A.GaussianBlur(blur_limit=7, p=1.0),
            ], p=0.3),
            
            # Noise variants
            A.OneOf([
                A.GaussNoise(std_range=(0.01, 0.05), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.05, 0.1), p=1.0),
            ], p=0.3),
            
            # Color adjustments
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),        
        ], 
    
        # Data augmentation parameters (randomly between +/- values)
        hsv_h=0.25,         # Hue adjustment (default 0.015)
        hsv_s=0.7,          # Saturation adjustment (default 0.7)
        hsv_v=0.6,          # Brightness adjustment (default 0.4)
        degrees=10,          # Rotation (default 0.0)
        translate=0.1,      # Translation (default 0.1)
        scale=0.5,          # Scaling (default 0.5
        shear = 0.0,        # Shearing (default 0.0)
        perspective=0.0,    # Perspective distortion (default 0.0)
        flipud=0.0,         # Vertical flip (default 0.0)
        fliplr=0.5,         # Horizontal flip (default 0.5)
        bgr=0.0,            # Random BGR distortion (default 0.0)
        mosaic=0.0,         # Mosaic augmentation (default 1.0)
        mixup=0,            # Mixup augmentation (default 0.0)
        cutmix=0.0,         # CutMix augmentation (default 0.0)
    )

    best_model_path = os.path.join(results.save_dir, "weights", "best.pt")
    print(f"Best model saved at: {best_model_path}")

    # inferece.py will use this best model for predictions.

    print("F1 score:", results.box.f1)
    print("Precision:", results.box.p)
    print("Recall:", results.box.r)
    print("mAP50:", results.box.map50)
