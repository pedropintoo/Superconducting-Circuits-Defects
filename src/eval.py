"""\
Simple evaluation script for YOLOv11 object detection models.

Requirements:
 - Install ../requirements.txt in a python virtual environment.
 - You need to have a trained model under the models/ directory. (defined by the 'best' variable)
 
Usage: python3 eval.py

Authors: Pedro Pinto, João Pinto, Fedor Chikhachev
"""
from ultralytics import YOLO

best = "models/best_model/weights/best.pt"
print(f"Using model: {best} for inference.")

model = YOLO(best)

model.val(
    data="dataset.yaml",
    imgsz=768,
    batch=32,
    save=True,
    project="inference_results",
    name=f"{best.split('/')[1]}_validation",
)
