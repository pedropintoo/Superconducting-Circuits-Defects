import os
from ultralytics import YOLO

# Load a pretrained model
model = YOLO("yolo11n.pt")

# Train the model on your custom dataset
results = model.train(
    data="chip_defects.yaml",
    epochs=125, 
    imgsz=640,
    project="chip_defect_detection",
    name="run",
    
# Data augmentation parameters (randomly between +/- values)
    hsv_h=0.5,          # Hue adjustment
    hsv_s=0.7,          # Saturation adjustment
    hsv_v=0.7,          # Brightness adjustment
    degrees=180,        # Rotation
    translate=0.4,      # Translation 
    scale=0.5,          # Scaling
    shear = 5,          # Shearing
    perspective=0.0,    # Perspective distortion
    flipud=0.5,         # Vertical flip
    fliplr=0.5,         # Horizontal flip
    bgr=0.5,            # Random BGR distortion
    mosaic=1.0,         # Mosaic augmentation
    mixup=0,            # Mixup augmentation
    cutmix=0.5,         # CutMix augmentation
)

best_model_path = os.path.join(results.save_dir, "weights", "best.pt")
print(f"Best model saved at: {best_model_path}")

# inferece.py will use this best model for predictions.
