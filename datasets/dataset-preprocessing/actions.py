"""Common preprocessing actions for YOLO/COCO datasets.

Expose reusable functions so individual scripts and the batch runner share the same logic.
"""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Iterable, List, Tuple

import yaml
from PIL import Image
from sahi.slicing import slice_coco as sahi_slice_coco

# -----------------
# Generic utilities
# -----------------
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_images(images_dir: Path) -> List[Path]:
    if not images_dir.exists():
        print(f"[WARN] list_images: missing {images_dir}")
        return []
    return sorted([p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS])


def copy_label(src_label: Path, dst_label: Path) -> None:
    if dst_label.resolve() == src_label.resolve():
        return
    if src_label.exists():
        shutil.copy2(src_label, dst_label)
    else:
        dst_label.write_text("", encoding="utf-8")


# -----------------
# YOLO -> COCO
# -----------------

def yolo_to_coco(
    yolo_root: Path,
    classes: dict[int, str],
    coco_out: Path,
    splits: Iterable[str] = ("train", "val"),
) -> Path:
    """Convert a YOLO dataset (images/labels) to COCO detection format."""

    categories = [
        {"id": int(cid), "name": cname, "supercategory": "none"}
        for cid, cname in classes.items()
    ]

    ensure_dir(coco_out)

    for split in splits:
        print(f"[COCO] YOLO -> COCO split '{split}' from {yolo_root} -> {coco_out}")
        images_dir = yolo_root / "images" / split
        labels_dir = yolo_root / "labels" / split
        export_dir = coco_out / split
        ensure_dir(export_dir)

        img_id = 1
        ann_id = 1
        coco = {"images": [], "annotations": [], "categories": categories}

        img_paths = list_images(images_dir)
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

            label_path = labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                lines = label_path.read_text(encoding="utf-8").strip().splitlines()
            else:
                lines = []

            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, xc, yc, w, h = parts
                cls_i = int(cls)
                xc_f, yc_f, w_f, h_f = map(float, (xc, yc, w, h))
                x_min = (xc_f - w_f / 2) * width
                y_min = (yc_f - h_f / 2) * height
                box_w = w_f * width
                box_h = h_f * height
                coco["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cls_i,
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
    print(f"[COCO] Wrote {len(coco['images'])} images, {len(coco['annotations'])} anns -> {out_json}")

    return coco_out


# -----------------
# COCO slicing
# -----------------

def slice_coco(
    coco_root: Path,
    sliced_out: Path,
    slice_height: int,
    slice_width: int,
    overlap_height_ratio: float,
    overlap_width_ratio: float,
    splits: Iterable[str]
) -> Path:
    """Slice COCO dataset using SAHI."""
    print(f"[SLICE] slice_height={slice_height}, slice_width={slice_width}, overlap={overlap_height_ratio}")
    annotation_name = "instances.json"
    for split in splits:
        coco_annotation_file_path = coco_root / split / annotation_name
        image_dir = coco_root / split
        output_dir = sliced_out / split
        ensure_dir(output_dir)

        coco_dict, _ = sahi_slice_coco(
            coco_annotation_file_path=str(coco_annotation_file_path),
            image_dir=str(image_dir),
            output_dir=str(output_dir),
            output_coco_annotation_file_name="instances", # In SAHI it will create instances_coco.json
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
        )
        print(f"[SLICE] Split '{split}': {len(coco_dict['images'])} tiles -> {output_dir/'instances_coco.json'}")
    return sliced_out


# -----------------
# COCO -> YOLO
# -----------------

def coco_to_yolo(
    coco_root: Path,
    yolo_out: Path,
    splits: Iterable[str]
) -> Path:
    """Convert COCO detection dataset back to YOLO format."""
    coco_json_name = "instances_coco.json"
    images_dir = yolo_out / "images"
    labels_dir = yolo_out / "labels"
    for split in splits:
        ensure_dir(images_dir / split)
        ensure_dir(labels_dir / split)

        instances_path = coco_root / split / coco_json_name
        
        if not instances_path.exists():
            print(f"[WARN] Missing {instances_path}, skipping split '{split}'")
            continue

        with instances_path.open("r", encoding="utf-8") as f:
            coco = json.load(f)

        images_by_id = {img["id"]: img for img in coco.get("images", [])}
        anns_by_image = {}
        for ann in coco.get("annotations", []):
            anns_by_image.setdefault(ann["image_id"], []).append(ann)

        img_count = 0
        ann_count = 0

        for img_id, img in images_by_id.items():
            file_name = img["file_name"]
            width = img["width"]
            height = img["height"]

            src_img = coco_root / split / file_name
            dst_img = images_dir / split / file_name
            if src_img.exists():
                shutil.copy2(src_img, dst_img)

            label_path = labels_dir / split / (Path(file_name).stem + ".txt")
            with label_path.open("w", encoding="utf-8") as lf:
                for ann in anns_by_image.get(img_id, []):
                    x, y, w, h = ann["bbox"]
                    x_c = (x + w / 2) / width
                    y_c = (y + h / 2) / height
                    w_n = w / width
                    h_n = h / height
                    lf.write(f"{ann['category_id']} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")
                    ann_count += 1
            img_count += 1
        print(f"[YOLO] Split '{split}': {img_count} images, {ann_count} anns -> {yolo_out}")
    return yolo_out


# -----------------
# Background balancing
# -----------------

def balance_backgrounds(
    yolo_root: Path,
    yolo_out: Path,
    target_bg_ratio: float,
    splits: Iterable[str],
    rng_seed: int,
) -> Path:
    random.seed(rng_seed)
    for split in splits:
        src_img_dir = yolo_root / "images" / split
        src_lbl_dir = yolo_root / "labels" / split
        dst_img_dir = yolo_out / "images" / split
        dst_lbl_dir = yolo_out / "labels" / split

        ensure_dir(dst_img_dir)
        ensure_dir(dst_lbl_dir)

        images = list_images(src_img_dir)
        bg_imgs: List[Tuple[Path, Path]] = []
        fg_imgs: List[Tuple[Path, Path]] = []
        for img in images:
            lbl = src_lbl_dir / f"{img.stem}.txt"
            if not lbl.exists() or not lbl.read_text(encoding="utf-8").strip():
                bg_imgs.append((img, lbl))
            else:
                fg_imgs.append((img, lbl))

        bg_count = len(bg_imgs)
        fg_count = len(fg_imgs)
        target_bg = int(round((target_bg_ratio / (1 - target_bg_ratio)) * fg_count))
        selected_bg = bg_imgs if bg_count <= target_bg else random.sample(bg_imgs, target_bg)

        print(
            f"[BG] Split '{split}': total {len(images)} | BG {bg_count} -> keep {len(selected_bg)} (~{target_bg_ratio*100:.1f}%)"
        )

        for img, lbl in fg_imgs + selected_bg:
            dst_img = dst_img_dir / img.name
            if dst_img.resolve() != img.resolve():
                shutil.copy2(img, dst_img)
            copy_label(lbl, dst_lbl_dir / f"{img.stem}.txt")

    classes_txt = yolo_root / "classes.txt"
    if classes_txt.exists():
        shutil.copy2(classes_txt, yolo_out / "classes.txt")
    return yolo_out


# -----------------
# Class balancing (downsample Dirt-Wire / class 1)
# -----------------

def balance_classes_downsample(
    yolo_root: Path,
    yolo_out: Path,
    critical_id: int,
    dirt_wire_id: int,
    splits: Iterable[str],
    rng_seed: int,
) -> Path:
    random.seed(rng_seed)

    for split in splits:
        src_img_dir = yolo_root / "images" / split
        src_lbl_dir = yolo_root / "labels" / split
        dst_img_dir = yolo_out / "images" / split
        dst_lbl_dir = yolo_out / "labels" / split

        ensure_dir(dst_img_dir)
        ensure_dir(dst_lbl_dir)

        critical_imgs = []
        dirt_wire_imgs = []
        background_imgs = []

        for img in list_images(src_img_dir):
            lbl = src_lbl_dir / f"{img.stem}.txt"
            if not lbl.exists():
                background_imgs.append(img)
                continue
            text = lbl.read_text(encoding="utf-8").strip()
            if not text:
                background_imgs.append(img)
                continue
            classes = {int(line.split()[0]) for line in text.splitlines() if line.split()}
            if critical_id in classes:
                critical_imgs.append(img)
            elif dirt_wire_id in classes:
                dirt_wire_imgs.append(img)
            else:
                background_imgs.append(img)

        critical_count = len(critical_imgs)
        dirt_wire_count = len(dirt_wire_imgs)

        selected_dirt_wire = dirt_wire_imgs
        if dirt_wire_count > critical_count:
            selected_dirt_wire = random.sample(dirt_wire_imgs, critical_count)

        print(
            f"[CLS-DOWN] Split '{split}': Critical {critical_count}, Dirt-Wire {dirt_wire_count}, BG {len(background_imgs)} -> keep Dirt-Wire {len(selected_dirt_wire)}"
        )

        for img in critical_imgs + selected_dirt_wire + background_imgs:
            shutil.copy2(img, dst_img_dir / img.name)
            lbl = src_lbl_dir / f"{img.stem}.txt"
            copy_label(lbl, dst_lbl_dir / f"{img.stem}.txt")

    classes_txt = yolo_root / "classes.txt"
    if classes_txt.exists():
        shutil.copy2(classes_txt, yolo_out / "classes.txt")
    return yolo_out


# -----------------
# Class balancing (hybrid upsample + downsample for Dirt-Wire)
# -----------------

def balance_classes_upsample(
    yolo_root: Path,
    yolo_out: Path,
    critical_id: int,
    dirt_wire_id: int,
    splits: Iterable[str],
    rng_seed: int,
) -> Path:
    random.seed(rng_seed)

    for split in splits:
        src_img_dir = yolo_root / "images" / split
        src_lbl_dir = yolo_root / "labels" / split
        dst_img_dir = yolo_out / "images" / split
        dst_lbl_dir = yolo_out / "labels" / split

        ensure_dir(dst_img_dir)
        ensure_dir(dst_lbl_dir)

        critical_imgs = []  # (path, c_count, nc_count)
        pure_nc_imgs = []   # (path, nc_count)
        background_imgs = []

        total_c = 0
        total_nc = 0

        for img in list_images(src_img_dir):
            lbl = src_lbl_dir / f"{img.stem}.txt"
            if not lbl.exists():
                background_imgs.append(img)
                continue
            text = lbl.read_text(encoding="utf-8").strip()
            if not text:
                background_imgs.append(img)
                continue
            c_count = 0
            nc_count = 0
            for line in text.splitlines():
                parts = line.split()
                if not parts:
                    continue
                cls_id = int(parts[0])
                if cls_id == critical_id:
                    c_count += 1
                elif cls_id == dirt_wire_id:
                    nc_count += 1
            if c_count > 0:
                critical_imgs.append((img, c_count, nc_count))
            elif nc_count > 0:
                pure_nc_imgs.append((img, nc_count))
            else:
                background_imgs.append(img)
            total_c += c_count
            total_nc += nc_count

        print(
            f"[CLS-UP] Split '{split}': Initial Critical {total_c}, Dirt-Wire {total_nc}, BG {len(background_imgs)}"
        )

        target = max(total_c, total_nc)
        needed_c = target - total_c
        extras_c = []
        if needed_c > 0 and critical_imgs:
            while needed_c > 0:
                choice = random.choice(critical_imgs)
                extras_c.append(choice)
                needed_c -= choice[1]
                total_c += choice[1]
                total_nc += choice[2]

        excess_nc = total_nc - total_c
        kept_pure_nc = list(pure_nc_imgs)
        if excess_nc > 0:
            random.shuffle(kept_pure_nc)
            current_excess = excess_nc
            filtered = []
            for path, nc_count in kept_pure_nc:
                if current_excess > 0:
                    current_excess -= nc_count
                else:
                    filtered.append((path, nc_count))
            kept_pure_nc = filtered
            total_nc -= excess_nc

        print(
            f"[CLS-UP] Split '{split}': Final Critical {total_c}, Dirt-Wire {total_nc}, BG {len(background_imgs)}"
        )

        # Copy outputs
        for img, _, _ in critical_imgs:
            shutil.copy2(img, dst_img_dir / img.name)
            copy_label(src_lbl_dir / f"{img.stem}.txt", dst_lbl_dir / f"{img.stem}.txt")

        for i, (img, _, _) in enumerate(extras_c):
            new_stem = f"{img.stem}_copy_{i}"
            shutil.copy2(img, dst_img_dir / f"{new_stem}{img.suffix}")
            copy_label(src_lbl_dir / f"{img.stem}.txt", dst_lbl_dir / f"{new_stem}.txt")

        for img, _ in kept_pure_nc:
            shutil.copy2(img, dst_img_dir / img.name)
            copy_label(src_lbl_dir / f"{img.stem}.txt", dst_lbl_dir / f"{img.stem}.txt")

        for img in background_imgs:
            shutil.copy2(img, dst_img_dir / img.name)
            copy_label(src_lbl_dir / f"{img.stem}.txt", dst_lbl_dir / f"{img.stem}.txt")

    classes_txt = yolo_root / "classes.txt"
    if classes_txt.exists():
        shutil.copy2(classes_txt, yolo_out / "classes.txt")
    return yolo_out
