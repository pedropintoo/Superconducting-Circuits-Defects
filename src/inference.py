import os
from ultralytics import YOLO

# ----------------------------------------------------------------
# Define which run to use for inference

RUN_ID = sorted(os.listdir("chip_defect_detection"))[-1]
EXAMPLE = "dark_open_circuit"

# ----------------------------------------------------------------

best = f"chip_defect_detection/{RUN_ID}/weights/best.pt"
examples = {
    "dark_big_burn": [f"../datasets/RQ3_TWPA_V2_W2/251023_Junctions/dark/{i:06d}.jpg" for i in range(280, 290)],
    "dark_open_circuit": [f"../datasets/RQ3_TWPA_V2_W2/251023_Junctions/dark/{i:06d}.jpg" for i in range(240, 250)],
    "dark_burn": [f"../datasets/RQ3_TWPA_V2_W2/251023_Junctions/dark/{i:06d}.jpg" for i in range(1199, 1211)],
}

model = YOLO(best)

results = model(
    examples[EXAMPLE],
    conf=0.20,
    save=True, 
    project="inference_results",
    name=f"{EXAMPLE}/{RUN_ID}",
    
)

print("Prediction completed and saved.")

for r in results:
    print(f"Processing result for image: {r.path}")
    