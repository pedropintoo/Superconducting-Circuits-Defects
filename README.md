# Superconducting Defect Detection (YOLO11 + SAHI)

Detecting microscopic fabrication defects on superconducting wafers. We target two defect types—Critical and Dirt-Wire—using a YOLO11 detector enhanced with a P2 head option and SAHI slicing to recover small-object recall. The pipeline includes label standardization (Label Studio), class/background rebalancing, SAHI tiling, and stitched full-image evaluation to approximate deployment conditions. Tested on Ubuntu 24.04 LTS (x86_64) in a VM with Python 3.12.3, Pip 24.0, and an NVIDIA GeForce RTX 4060 (8 GB VRAM).


## Results Snapshot

**Metrics (val split)**

| Configuration | Critical P | Critical R | Critical F1 | Dirt-Wire P | Dirt-Wire R | Dirt-Wire F1 | mAP50 | mAP50:95 | F1_mean | Time (min) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline (Full Image) | 0.302 | 0.174 | 0.221 | 0.000 | 0.000 | 0.000 | 0.000 | 0.024 | 0.110 | 34.4 |
| Baseline + Augmentations | 0.403 | 0.203 | 0.273 | 0.000 | 0.000 | 0.000 | 0.013 | 0.038 | 0.137 | 24.5 |
| Baseline + P2 Head | 0.410 | 0.130 | 0.198 | 0.365 | 0.041 | 0.074 | 0.087 | 0.017 | 0.136 | 40.7 |
| SAHI 128px + 50% BG | 0.243 | 0.391 | 0.300 | 0.284 | 0.489 | 0.360 | 0.243 | 0.101 | 0.330 | 43.9 |
| SAHI 256px + 50% BG (best) | 0.193 | 0.503 | 0.435 | 0.674 | 0.560 | 0.612 | 0.368 | 0.152 | 0.497 | 155.1 |
| SAHI 256px DS + 50% BG | 0.364 | 0.304 | 0.330 | 0.525 | 0.323 | 0.400 | 0.235 | 0.105 | 0.365 | 19.2 |

**Visual examples**
- Defect detection prediction (Critical in red, Dirt-Wire in cyan): ![Prediction example](docs/assets/prediction_example.png)

## Quickstart
1) **Set up env**
```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

2) **Preprocess data (tiling + balancing)**
```bash
cd datasets/dataset-preprocessing
python3 run_pipeline.py \
	--actions prepare_labels_and_split yolo_to_coco slice_coco coco_to_yolo balance_classes_upsample balance_backgrounds \
	--labels-root ../labeling_output \
	--images-root ../full_dataset \
	--split-ratios 0.8 0.2 \
	--slice-height 256 --slice-width 256 --slice-overlap 0.2 \
	--background-target 0.50 \
	--output-dir ../processed_dataset \
	--clean-targets
cd ../..
```

3) **Train (YOLO11 + SAHI-ready slices)**
```bash
cd src
python3 train.py
```
Outputs under [models/<run>/weights/](models) (YOLO run folders); default best at [models/best_model/weights/best.pt](models/best_model/weights/best.pt).

4) **Evaluate**
- YOLO eval: `python3 eval.py --weights models/best_model/weights/best.pt`
- SAHI eval (global metrics):
```bash
python3 eval-sahi.py \
	--weights models/best_model/weights/best.pt \
	--split val \
	--data_root ../datasets/train_val_dataset
```

5) **Inference**
- CLI sliced inference demo: `python3 inference.py`
- Streamlit UI: `streamlit run inference-web.py`

## Data & NDA note
We provide an NDA-compliant subset: [datasets/full_dataset](datasets/full_dataset) (images) and [datasets/labeling_output](datasets/labeling_output) (annotations). Processed datasets live in [datasets/processed_dataset](datasets/processed_dataset) and [datasets/train_val_dataset](datasets/train_val_dataset) after running the pipeline. Raw proprietary data beyond this subset is not distributed.

## Repository map
- Data: [datasets](datasets/README.md)
- Preprocessing pipeline: [datasets/dataset-preprocessing](datasets/dataset-preprocessing/README.md)
- Labeling workflow + ML backend: [labeling](labeling/README.md) and [labeling/model_backend](labeling/model_backend/README.md)
- Training / eval / inference: [src](src/README.md)
- Methodology notes: [docs/methodology.md](docs/methodology.md)

## Problem statement (high level)
Goal: detect superconducting fabrication defects (Critical, Dirt-Wire) in wafer imagery. The model uses YOLO11 with optional P2 head and SAHI slicing to improve small-defect recall. We balance background tiles, upsample minority classes, and evaluate on stitched full-image metrics to approximate real-world deployment.