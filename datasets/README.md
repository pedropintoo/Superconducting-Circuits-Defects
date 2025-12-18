# Datasets

We now ship an NDA-compliant subset so you can run the full pipeline end-to-end.

## What’s here
- [full_dataset/](full_dataset): raw images (subset released under NDA constraints).
- [labeling_output/](labeling_output): YOLO-format annotations exported from Label Studio and aligned to [full_dataset/](full_dataset).
- [processed_dataset/](processed_dataset): SAHI-sliced YOLO dataset produced by the pipeline (train/val/test under images/ and labels/).
- [train_val_dataset/](train_val_dataset): convenience copy for training/eval once the pipeline finishes.

## How these are produced
Run the preprocessing pipeline in [dataset-preprocessing/](dataset-preprocessing) to:
1) Map labels to images and split (train/val[/test]).
2) Convert YOLO → COCO for slicing.
3) SAHI slice (tiling) with overlap.
4) Convert back to YOLO.
5) Balance classes (upsample) and background ratio.
6) Write the final dataset to [processed_dataset/](processed_dataset) and copy to [train_val_dataset/](train_val_dataset).

Example (from inside [dataset-preprocessing/](dataset-preprocessing)):
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

## NDA note
Only a portion of the real-world data is included here due to NDA. The pipeline and code are fully reproducible on the provided subset; swap in your own data under the same folder structure to retrain.

