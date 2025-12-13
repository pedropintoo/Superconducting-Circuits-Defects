## One-line preprocessing

Run the full pipeline: map labels to images, convert to COCO, slice, convert back to YOLO, upsample classes, and balance backgrounds.

```bash
python3 run_pipeline.py \
  --actions prepare_labels_and_split yolo_to_coco slice_coco coco_to_yolo balance_classes_upsample balance_backgrounds \
  --labels-root ../labeling_output \
  --images-root ../full_dataset \
  --split-ratios 0.8 0.2 \
  --slice-height 256 --slice-width 256 --slice-overlap 0.2 \
  --background-target 0.50 \
  --output-dir ../processed_dataset \
  --clean-targets
```

In this example we:
- Use labels from `../labeling_output` and images from `../full_dataset`.
- Split into 80% train / 20% val.
- Convert labels to COCO format for slicing.
- Slice into 256x256 tiles with 20% overlap.
- Convert back to YOLO format.
- Upsample minority classes to match the majority class count.
- Balance backgrounds to achieve ~50% background tiles.
- Write the final dataset to `../processed_dataset`.
- Clean up intermediate files.

## Detailed explanation

Key flags (just the essentials):
- `--split-ratios`: two values → train/val; three values → train/val/test.
- `--slice-height/--slice-width/--slice-overlap`: SAHI tiling size and overlap.
- `--background-target`: target background ratio after balancing (e.g., 0.50 ≈ 1:1 BG vs FG).
- `--output-dir`: where the final YOLO dataset is written.
- `--clean-targets`: remove intermediates at the end.

## What's in this folder
- `run_pipeline.py`: orchestrates the steps above.
- `actions.py`: implementations for prepare split, YOLO↔COCO, slicing, class/background balancing.
- `convert_to_unique.sh`, `count_train_classes.py`, `see_image.py`: small utilities for dataset inspection/debug.
- Intermediates are written under `tmp_pipeline/` (auto-cleaned when `--clean-targets` is set); final data goes to `--output-dir`.
