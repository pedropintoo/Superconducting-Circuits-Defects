import os
from ultralytics import YOLO

# Load a pretrained model
model = YOLO("yolo11s.pt")

# Train the model on your custom dataset
results = model.train(
    data="dataset.yaml",
    epochs=500, 
    imgsz=1280,
    batch=4,
    project="chip_defect_detection",
    name="run",
    
# Data augmentation parameters (randomly between +/- values)
    hsv_h=0.25,         # Hue adjustment (default 0.015)
    hsv_s=0.7,          # Saturation adjustment (default 0.7)
    hsv_v=0.6,          # Brightness adjustment (default 0.4)
    degrees=0,          # Rotation (default 0.0)
    translate=0.4,      # Translation (default 0.1)
    scale=0.5,          # Scaling (default 0.5)
    shear = 0.0,        # Shearing (default 0.0)
    perspective=0.0,    # Perspective distortion (default 0.0)
    flipud=0.0,         # Vertical flip (default 0.0)
    fliplr=0.5,         # Horizontal flip (default 0.5)
    bgr=0.0,            # Random BGR distortion (default 0.0)
    mosaic=1.0,         # Mosaic augmentation (default 1.0)
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