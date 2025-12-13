python run_pipeline.py \
    --actions yolo_to_coco slice_coco coco_to_yolo balance_classes_upsample balance_backgrounds \
    --yolo-root ../new_train_val_dataset \
    --slice-height 256 --slice-width 256 \
    --background-target 0.20
 