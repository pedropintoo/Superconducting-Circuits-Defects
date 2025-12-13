import os
from ultralytics import YOLO
from ultralytics import settings
import albumentations as A
import cv2

settings.update({"tensorboard": True})

trainings = [
    ("new_dataset_sliced_256_balanced_upsampled_bg50_augmentation_without_mosaic_imgsz_768_using_p2_head", -1, 768, "yolo11n", 0.25, 0.7, 0.6, 0.5, True),
]

for name, batch, imgsz, yolo_version, hsv_h, hsv_s, hsv_v, scale, extra_aug in trainings:

    # Using P2-head
    model = YOLO("yolo11n-p2.yaml").load(f'{yolo_version}.pt')
    
    # model = YOLO(f"{yolo_version}.pt")

    results = model.train(
        data="dataset.yaml",
        epochs=200, 
        imgsz=imgsz,
        batch=batch,
        project="chip_defect_detection",
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
        ] if extra_aug else [], 
    
        # Data augmentation parameters (randomly between +/- values)
        hsv_h=hsv_h,         # Hue adjustment (default 0.015)
        hsv_s=hsv_s,          # Saturation adjustment (default 0.7)
        hsv_v=hsv_v,          # Brightness adjustment (default 0.4)
        degrees=10,          # Rotation (default 0.0)
        translate=0.1,      # Translation (default 0.1)
        scale=scale,          # Scaling (default 0.5)                 # use 0 when tiling is off!
        shear = 0.0,        # Shearing (default 0.0)
        perspective=0.0,    # Perspective distortion (default 0.0)
        flipud=0.0,         # Vertical flip (default 0.0)
        fliplr=0.5,         # Horizontal flip (default 0.5)
        bgr=0.0,            # Random BGR distortion (default 0.0)
        mosaic=0.0,         # Mosaic augmentation (default 1.0)
        mixup=0,            # Mixup augmentation (default 0.0)
        cutmix=0.0,         # CutMix augmentation (default 0.0)
        
        ## REMOVE THIS LINE FOR REAL TRAINING!
        # close_mosaic=0,
    )

    best_model_path = os.path.join(results.save_dir, "weights", "best.pt")
    print(f"Best model saved at: {best_model_path}")

    # inferece.py will use this best model for predictions.

    print("F1 score:", results.box.f1)
    print("Precision:", results.box.p)
    print("Recall:", results.box.r)
    print("mAP50:", results.box.map50)
