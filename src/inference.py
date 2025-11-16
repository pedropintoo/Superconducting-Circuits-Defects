import os
from ultralytics import YOLO

# ----------------------------------------------------------------
# Define which run to use for inference

all_models = sorted(
    [m.split("run")[-1] or "0" for m in os.listdir("chip_defect_detection")],
    key=int
)
all_models = [m if m != "0" else "" for m in all_models]

RUN_ID = "run"+all_models[-1]  # Use the latest run
EXAMPLE = "dark_big_burn"
# ----------------------------------------------------------------

best = f"chip_defect_detection/{RUN_ID}/weights/best.pt"
print(f"Using model: {best} for inference.")

examples = {
    "dark_big_burn": [f"../datasets/RQ3_TWPA_V2_W2/251023_Junctions/dark/{i:06d}.jpg" for i in range(280, 290)],
    "dark_open_circuit": [f"../datasets/RQ3_TWPA_V2_W2/251023_Junctions/dark/{i:06d}.jpg" for i in range(240, 250)],
    "dark_burn": [f"../datasets/RQ3_TWPA_V2_W2/251023_Junctions/dark/{i:06d}.jpg" for i in range(1199, 1211)],
    "chips_1": [f"../datasets/RQ3_TWPA_V2_W2/251101_Chips/v2_7500_500_DF/{i:06d}.jpg" for i in range(50, 60)],
    "dark_2": [f"../datasets/RQ3_TWPA_V2_W2/251023_Junctions/dark/{i:06d}.jpg" for i in range(469, 500)],
    "white_chips": [f"../datasets/RQ3_TWPA_V2_W2/251101_Chips/v2_7500_500/{i:06d}.jpg" for i in range(15, 30)],
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
    