import os
from ultralytics import YOLO

run_id = "run1"
best = f"chip_defect_detection/{run_id}/weights/best.pt"
image_path = "../data/RQ3_TWPA_V2_W2/251023_Junctions/dark"
test_range = range(280, 290)
images = [f"{image_path}/{i:06d}.jpg" for i in test_range]

model = YOLO(best)

results = model(
    images,
    conf=0.05,
    save=True
)

print("Prediction completed and saved.")

for r in results:
    print(f"Processing result for image: {r.path}")
    print(f"Boxes: {r.boxes}")  # Boxes object for bounding box outputs
    print(f"Masks: {r.masks}")  # Masks object for segmentation masks outputs
    print(f"Keypoints: {r.keypoints}")  # Keypoints object for pose