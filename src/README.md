# Source Guide

Entrypoints for training, evaluation, and inference. Default weights: [models/best_model/weights/best.pt](models/best_model/weights/best.pt).

## Training
- [train.py](train.py): Trains YOLO11 on SAHI-sliced data ([../datasets/processed_dataset](../datasets/processed_dataset) via [dataset.yaml](dataset.yaml)). Default run logged under [models/<run>/](../models); best weights saved to [models/best_model/weights/best.pt](models/best_model/weights/best.pt).
- Configs: [yolo11.yaml](yolo11.yaml) (base), [yolo11-p2.yaml](yolo11-p2.yaml) (optional P2 head for small objects).

Run:
```bash
python3 train.py
```

## Evaluation
- [eval.py](eval.py): Standard YOLO validation on [dataset.yaml](dataset.yaml) (unsliced metrics).
    ```bash
    python3 eval.py --weights models/best_model/weights/best.pt
    ```
- [eval-sahi.py](eval-sahi.py): YOLO + SAHI stitched full-image eval (global metrics, confusion, visuals).
    ```bash
    python3 eval-sahi.py \
        --weights models/best_model/weights/best.pt \
        --split val \
        --data_root ../datasets/train_val_dataset
    ```
Outputs: [inference_results_sahi/exp*/](inference_results_sahi) with metrics.txt, confusion_matrix_sahi.png, visuals/, coco_gt.json, coco_dt.json.

## Inference
- [inference.py](inference.py): CLI demo with SAHI slicing over sample images; writes visuals under [demo_data/](demo_data).
- [inference-web.py](inference-web.py): Streamlit app; choose weights from models/*/weights/best.pt, upload images/zip or use demos; supports slicing params and confidence threshold.
    ```bash
    streamlit run inference-web.py
    ```

## Data config
- [dataset.yaml](dataset.yaml): Points to [../datasets/processed_dataset](../datasets/processed_dataset) (train/val/test) with classes {0: Critical, 1: Dirt-Wire}.
- yolo11*.yaml: Model definitions (base and P2 variant).

## Outputs & defaults
- Models: [models/<run>/weights/best.pt](../models) (default referenced as best_model).
- Eval: [runs/detect/](runs/detect) (YOLO) and [inference_results_sahi/exp*](inference_results_sahi) (SAHI stitched eval).
- Inference visuals: [demo_data/prediction_visual_*.png](demo_data).

