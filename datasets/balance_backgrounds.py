"""
Balance background images in a YOLO dataset to a target percentage (default 8%).
- Counts how many background images (empty/missing label) exist per split.
- Writes a new dataset with background proportion near target, downsampling backgrounds if needed.
- Source is `train_val_dataset`, output is `train_val_dataset_bg8` (non-destructive).
"""
import random
import shutil
from pathlib import Path

TARGET_BG_RATIO = 0.08  # 8% backgrounds
RNG_SEED = 42  # deterministic sampling

ROOT = Path(__file__).resolve().parent
SRC_ROOT = ROOT / "train_val_dataset_sliced_balanced_upsampled"
DST_ROOT = ROOT / "train_val_dataset_sliced_balanced_upsampled_bg8"
SPLITS = ["train", "val"]
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

random.seed(RNG_SEED)


def list_images(images_dir: Path):
    return sorted([p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS])


def is_background(label_path: Path) -> bool:
    if not label_path.exists():
        return True
    text = label_path.read_text(encoding="utf-8").strip()
    return len(text) == 0


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def copy_label_or_empty(src_label: Path, dst_label: Path):
    if src_label.exists():
        shutil.copy2(src_label, dst_label)
    else:
        dst_label.write_text("", encoding="utf-8")


def balance_split(split: str):
    src_img_dir = SRC_ROOT / "images" / split
    src_lbl_dir = SRC_ROOT / "labels" / split
    dst_img_dir = DST_ROOT / "images" / split
    dst_lbl_dir = DST_ROOT / "labels" / split

    ensure_dir(dst_img_dir)
    ensure_dir(dst_lbl_dir)

    images = list_images(src_img_dir)
    if not images:
        print(f"[WARN] No images for split '{split}' in {src_img_dir}")
        return

    bg_imgs = []
    fg_imgs = []
    for img in images:
        lbl = src_lbl_dir / f"{img.stem}.txt"
        if is_background(lbl):
            bg_imgs.append((img, lbl))
        else:
            fg_imgs.append((img, lbl))

    total = len(images)
    bg_count = len(bg_imgs)
    fg_count = len(fg_imgs)
    current_ratio = bg_count / total if total else 0.0

    # target_bg = (r/(1-r)) * fg_count achieves bg/(bg+fg)=r
    target_bg = int(round((TARGET_BG_RATIO / (1 - TARGET_BG_RATIO)) * fg_count))
    selected_bg = bg_imgs
    if bg_count > target_bg:
        selected_bg = random.sample(bg_imgs, target_bg)
    # if we have fewer backgrounds than needed, we just keep all we have

    final_bg = len(selected_bg)
    final_total = fg_count + final_bg
    final_ratio = final_bg / final_total if final_total else 0.0

    print(
        f"Split '{split}': {total} imgs | BG {bg_count} ({current_ratio*100:.2f}%) -> keeping {final_bg} ({final_ratio*100:.2f}%)"
    )

    for img, lbl in fg_imgs:
        shutil.copy2(img, dst_img_dir / img.name)
        dst_lbl = dst_lbl_dir / f"{img.stem}.txt"
        copy_label_or_empty(lbl, dst_lbl)

    for img, lbl in selected_bg:
        shutil.copy2(img, dst_img_dir / img.name)
        dst_lbl = dst_lbl_dir / f"{img.stem}.txt"
        copy_label_or_empty(lbl, dst_lbl)


if __name__ == "__main__":
    print("Counting backgrounds and generating dataset with ~8% background...")
    DST_ROOT.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        balance_split(split)
    # copy classes.txt if exist
    classes_txt = SRC_ROOT / "classes.txt"
    if classes_txt.exists():
        shutil.copy2(classes_txt, DST_ROOT / "classes.txt")
    print(f"\nDone. New dataset at {DST_ROOT}")

