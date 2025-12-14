# SAHI Inference & Evaluation Script

This script (`eval-sahi.py`) runs **YOLO + SAHI inference on full-resolution images**. It merges sliced predictions back into global coordinates, compares them against ground-truth labels, and computes real-world performance metrics, not just slice-level metrics.

## Features

* **Full Image Evaluation**: Calculates metrics on global images, not just slices.
* **Metrics**: Outputs mAP (COCO), Precision, Recall, F1-Score, and Confusion Matrix.
* **Visualization**: Saves images with Ground Truth (Green) and Predictions (Red) drawn for visual verification.
* **Experiment Tracking**: Automatically saves results to unique folders (`inference_results_sahi/expN`).

## Usage

Run the script from the terminal.

Example command:

```bash
python eval-sahi.py \
    --weights models/best_model/weights/best.pt \
    --split val \
    --data_root ../datasets/train_val_dataset
```

### Arguments

* `--weights`: Path to your trained YOLO `.pt` model.
* `--split`: Dataset split to evaluate (e.g., `val`, `train`, `test`). Default: `val`.
* `--data_root`: Path to the dataset root folder (containing `images/` and `labels/`).
* `--project`: Base folder for saving results. Default: `inference_results_sahi`.
* `--name`: Name of the experiment subfolder. Default: `exp`.

## Output

Results are saved in `inference_results_sahi/expN/` and include:

* `metrics.txt`: Summary of mAP, F1, Precision, and Recall.
* `confusion_matrix_sahi.png`: Visual confusion matrix.
* `visuals/`: Images with plotted bounding boxes (GT vs. Prediction).
* `coco_gt.json` / `coco_dt.json`: Raw COCO-format data files.

