"""\
Simple evaluation script for YOLOv11 object detection models.

Authors: Pedro Pinto, João Pinto, Fedor Chikhachev
"""
from ultralytics import YOLO

best = "models/new_dataset_sliced_256_balanced_upsampled_bg50_augmentation_without_mosaic_imgsz_768_2/weights/best.pt"
print(f"Using model: {best} for inference.")

model = YOLO(best)

model.val(
    data="new_128_dataset_sliced_balanced_upsampled_bg50.yaml",
    imgsz=768,
    batch=-1,
    save=True,
    project="inference_results",
    name=f"{best.split('/')[1]}_validation",
)
