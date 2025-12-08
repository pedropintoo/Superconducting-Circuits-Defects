# convert_coco_to_yolo.py
import json
import shutil
from collections import defaultdict
from pathlib import Path

# Paths anchored to this script
ROOT = Path(__file__).resolve().parent

# Input COCO dataset (expects subfolders train/val with instances.json and images)
COCO_ROOT_DIR = ROOT / "coco_sliced"

# Output YOLO dataset
YOLO_OUT_ROOT = ROOT / "train_val_dataset_sliced"
YOLO_IMAGES_DIR = YOLO_OUT_ROOT / "images"
YOLO_LABELS_DIR = YOLO_OUT_ROOT / "labels"

SPLITS = ["train", "val"]


def coco_bbox_to_yolo(bbox, img_w, img_h):
    x, y, w, h = bbox
    x_c = (x + w / 2) / img_w
    y_c = (y + h / 2) / img_h
    w_n = w / img_w
    h_n = h / img_h
    return x_c, y_c, w_n, h_n


def ensure_dirs(split: str):
    (YOLO_IMAGES_DIR / split).mkdir(parents=True, exist_ok=True)
    (YOLO_LABELS_DIR / split).mkdir(parents=True, exist_ok=True)


def load_coco(instances_path: Path):
    with instances_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def export_split(split: str):
    instances_path = COCO_ROOT_DIR / split / "instances_coco.json"
    if not instances_path.exists():
        print(f"[WARN] instances.json not found for split '{split}' in {instances_path}")
        return

    ensure_dirs(split)
    coco = load_coco(instances_path)

    images_by_id = {img["id"]: img for img in coco.get("images", [])}
    anns_by_image = defaultdict(list)
    for ann in coco.get("annotations", []):
        anns_by_image[ann["image_id"]].append(ann)

    img_count = 0
    ann_count = 0

    for img_id, img in images_by_id.items():
        file_name = img["file_name"]
        width = img["width"]
        height = img["height"]

        src_img = COCO_ROOT_DIR / split / file_name
        dst_img = YOLO_IMAGES_DIR / split / file_name
        if src_img.exists():
            shutil.copy2(src_img, dst_img)

        label_path = YOLO_LABELS_DIR / split / (Path(file_name).stem + ".txt")
        with label_path.open("w", encoding="utf-8") as lf:
            for ann in anns_by_image.get(img_id, []):
                x_c, y_c, w_n, h_n = coco_bbox_to_yolo(ann["bbox"], width, height)
                lf.write(f"{ann['category_id']} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")
                ann_count += 1

        img_count += 1

    print(
        f"Split '{split}': {img_count} copied images, {ann_count} annotations exported."
    )


def main():
    print("Inicializing COCO to YOLO conversion...")
    YOLO_OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        export_split(split)
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
