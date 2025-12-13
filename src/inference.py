import os
from ultralytics import YOLO
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel
import cv2
import numpy as np

# ----------------------------------------------------------------
# Define which run to use for inference

# all_models = sorted(
#     [m.split("run")[-1] or "0" for m in os.listdir("chip_defect_detection")],
#     key=int
# )
# all_models = [m if m != "0" else "" for m in all_models]

# RUN_ID = "run"+all_models[-1]  # Use the latest run
EXAMPLE = "LO_mark"
# ----------------------------------------------------------------

best = "chip_defect_detection/new_dataset_sliced_128_balanced_upsampled_bg20_augmentation_without_mosaic_imgsz_7682/weights/best.pt"
print(f"Using model: {best} for inference.")

examples = {
    "dark_big_burn": [f"../datasets/RQ3_TWPA_V2_W2/251023_Junctions/dark/{i:06d}.jpg" for i in range(280, 290)],
    "dark_open_circuit": [f"../datasets/RQ3_TWPA_V2_W2/251023_Junctions/dark/{i:06d}.jpg" for i in range(240, 250)],
    "dark_burn": [f"../datasets/RQ3_TWPA_V2_W2/251023_Junctions/dark/{i:06d}.jpg" for i in range(1199, 1211)],
    "chips_1": [f"../datasets/RQ3_TWPA_V2_W2/251101_Chips/v2_7500_500_DF/{i:06d}.jpg" for i in range(50, 60)],
    "dark_2": [f"../datasets/RQ3_TWPA_V2_W2/251023_Junctions/dark/{i:06d}.jpg" for i in range(469, 500)],
    "white_chips": [f"../datasets/RQ3_TWPA_V2_W2/251101_Chips/v2_7500_500/{i:06d}.jpg" for i in range(15, 30)],
    "spir_60": [f"../datasets/full_dataset/Second_Batch-PM251015p1-251028_Spir_60_6-bright-{i:06d}.jpg" for i in range(220, 230)],
    "LO_mark": [f"../datasets/full_dataset/Second_Batch-PM251015p1-251022_post_LO_mark-dark-{i:06d}.jpg" for i in range(342, 350)],
    "Val_examples": [f"../datasets/full_dataset/RQ3_TWPA_V2_W2-251023_Junctions-dark-00015{i:01d}.jpg" for i in range(9)],
    "random" : ["../datasets/full_dataset/Second_Batch-PM251015p1-251022_post_LO_mark-dark-000169.jpg"]
}

model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=best,
    confidence_threshold=0.5,
    device="cpu", 
)

results = []
for image_path in examples[EXAMPLE][1:4 ]:  # Process all examples
    t0 = cv2.getTickCount()
    result = get_sliced_prediction(
        image_path,
        model,
        slice_height=128,
        slice_width=128,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
    )
    t1 = cv2.getTickCount()
    time_taken = (t1 - t0) / cv2.getTickFrequency()
    print(f"Time taken for sliced inference on {image_path}: {time_taken:.3f} seconds")
    results.append((image_path, result))

    result.export_visuals(export_dir="demo_data/")
    # move to a unique name
    os.rename(
        "demo_data/prediction_visual.png",
        f"demo_data/prediction_visual_{os.path.basename(image_path)}.png",
    )
    print(f"Processed and saved results for image: {image_path}")

# CV2 to each predict
for image_path, result in results:
    result_image = cv2.imread(f"demo_data/prediction_visual_{os.path.basename(image_path)}.png")
    result_image = cv2.resize(result_image, (1200, 800))  # Resize for better visibility
    cv2.imshow(f"Prediction for {os.path.basename(image_path)}", result_image)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()


# best = "chip_defect_detection/sliced_640/weights/best.pt"
# model = YOLO(best)

# results = model(
#     examples[EXAMPLE],
#     conf=0.20,
#     save=True, 
#     project="inference_results",
#     name=f"{best.split('/')[1]}_{EXAMPLE}",
    
# )

# print("Prediction completed and saved.")

# for r in results:
#     print(f"Processing result for image: {r.path}")

# EVALUATION
# best = "chip_defect_detection/balanced_downsampled_bg8_2/weights/best.pt"
# model = YOLO(best)

# model.val(
#     data="new_128_dataset_sliced_balanced_upsampled_bg8.yaml",
#     imgsz=500,
#     batch=8,
#     save=True,
#     project="inference_results",
#     name=f"{best.split('/')[1]}_validation",
#     # device="cpu"
# )