# convert_yolo_to_coco.py
import json
import shutil
from pathlib import Path

import yaml
from PIL import Image

# Base paths anchored to this file so it works no matter where you run it
ROOT = Path(__file__).resolve().parent

DATASET_ROOT_DIR = ROOT / "train_val_dataset"

YOLO_IMAGES_DIR = DATASET_ROOT_DIR / "images"
YOLO_LABELS_DIR = DATASET_ROOT_DIR / "labels"

YOLO_DATA_YAML = ROOT.parent / "src" / "dataset.yaml" 

COCO_OUTPUT_DIR = ROOT / "coco_dataset"  

COCO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================================
# 2. Load and parse YOLO dataset config
# =========================================================================
print("Starting YOLO to COCO conversion...")

with open(YOLO_DATA_YAML, "r", encoding="utf-8") as f:
    yolo_cfg = yaml.safe_load(f)

classes = yolo_cfg.get("names") or {}
categories = [
    {"id": int(cid), "name": cname, "supercategory": "none"}
    for cid, cname in classes.items()
]

splits_to_export = ["train", "val"]
print(f"Splits found in config: {splits_to_export}")

# =========================================================================
# 3. Export each split
# =========================================================================

def yolo_label_file(img_path: Path, labels_dir: Path) -> Path:
    return labels_dir / (img_path.stem + ".txt")


def load_yolo_annotations(label_path: Path):
    if not label_path.exists():
        return []
    anns = []
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, x_center, y_center, w, h = parts
            anns.append(
                (
                    int(cls),
                    float(x_center),
                    float(y_center),
                    float(w),
                    float(h),
                )
            )
    return anns


def export_split(split_name: str):
    images_dir = YOLO_IMAGES_DIR / split_name
    labels_dir = YOLO_LABELS_DIR / split_name
    export_dir = COCO_OUTPUT_DIR / split_name
    export_dir.mkdir(parents=True, exist_ok=True)

    img_id = 1
    ann_id = 1
    coco = {"images": [], "annotations": [], "categories": categories}

    img_paths = sorted(
        [
            *images_dir.glob("*.jpg"),
            *images_dir.glob("*.jpeg"),
            *images_dir.glob("*.png"),
            *images_dir.glob("*.bmp"),
            *images_dir.glob("*.tif"),
            *images_dir.glob("*.tiff"),
        ]
    )

    print(f"\n- Exporting split '{split_name}' with {len(img_paths)} images...")

    for img_path in img_paths:
        with Image.open(img_path) as im:
            width, height = im.size

        coco["images"].append(
            {
                "id": img_id,
                "file_name": img_path.name,
                "width": width,
                "height": height,
            }
        )

        label_path = yolo_label_file(img_path, labels_dir)
        anns = load_yolo_annotations(label_path)

        for cls, xc, yc, w, h in anns:
            x_min = (xc - w / 2) * width
            y_min = (yc - h / 2) * height
            box_w = w * width
            box_h = h * height
            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls,
                    "bbox": [x_min, y_min, box_w, box_h],
                    "area": box_w * box_h,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

        shutil.copy2(img_path, export_dir / img_path.name)
        img_id += 1

    out_json = export_dir / "instances.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)

    print(
        f"Split '{split_name}': {len(coco['images'])} images, {len(coco['annotations'])} annotations exported."
    )


for split_name in splits_to_export:
    export_split(split_name)

print("\n✅ Done!")