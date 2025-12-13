"""
Balance classes in a YOLO dataset by downsampling the Non-Critical class.
- Counts Critical (class 0) and Non-Critical (class 1) samples per split.
- Removes Non-Critical samples until they match the number of Critical samples.
- Source is `train_val_dataset_sliced_bg8`, output is `train_val_dataset_sliced_bg8_balanced`.
"""
import random
import shutil
from pathlib import Path
from collections import defaultdict

RNG_SEED = 42  # deterministic sampling

ROOT = Path(__file__).resolve().parent
SRC_ROOT = ROOT / "train_val_dataset_sliced_bg8"
DST_ROOT = ROOT / "train_val_dataset_sliced_bg8_balanced"
SPLITS = ["train", "val"]
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

CRITICAL_CLASS = 0
NON_CRITICAL_CLASS = 1

random.seed(RNG_SEED)


def list_images(images_dir: Path):
    return sorted([p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS])


def get_classes_in_label(label_path: Path) -> set:
    """Return set of class IDs in a label file. Empty set if no label or background."""
    if not label_path.exists():
        return set()
    classes = set()
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return set()
    for line in text.split("\n"):
        parts = line.strip().split()
        if parts:
            try:
                cls = int(parts[0])
                classes.add(cls)
            except ValueError:
                pass
    return classes


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def copy_image_and_label(src_img: Path, src_lbl_dir: Path, dst_img_dir: Path, dst_lbl_dir: Path):
    """Copy image and its label file."""
    shutil.copy2(src_img, dst_img_dir / src_img.name)
    src_lbl = src_lbl_dir / f"{src_img.stem}.txt"
    dst_lbl = dst_lbl_dir / f"{src_img.stem}.txt"
    if src_lbl.exists():
        shutil.copy2(src_lbl, dst_lbl)
    else:
        dst_lbl.write_text("", encoding="utf-8")


def balance_split(split: str):
    src_img_dir = SRC_ROOT / "images" / split
    src_lbl_dir = SRC_ROOT / "labels" / split
    dst_img_dir = DST_ROOT / "images" / split
    dst_lbl_dir = DST_ROOT / "labels" / split

    ensure_dir(dst_img_dir)
    ensure_dir(dst_lbl_dir)

    images = list_images(src_img_dir)
    if not images:
        print(f"[WARN] Sem imagens para split '{split}' em {src_img_dir}")
        return

    # Categorize images by what classes they contain
    critical_imgs = []      # contain class 0 (Critical)
    non_critical_imgs = []  # contain class 1 (Non-Critical) but NOT class 0
    background_imgs = []    # no annotations (background)

    for img in images:
        lbl = src_lbl_dir / f"{img.stem}.txt"
        classes = get_classes_in_label(lbl)

        if not classes:
            background_imgs.append(img)
        elif CRITICAL_CLASS in classes:
            critical_imgs.append(img)
        elif NON_CRITICAL_CLASS in classes:
            non_critical_imgs.append(img)

    critical_count = len(critical_imgs)
    non_critical_count = len(non_critical_imgs)
    bg_count = len(background_imgs)
    total = critical_count + non_critical_count + bg_count

    # Downsample Non-Critical to match Critical count
    selected_non_critical = non_critical_imgs
    if non_critical_count > critical_count:
        selected_non_critical = random.sample(non_critical_imgs, critical_count)

    final_critical = len(critical_imgs)
    final_non_critical = len(selected_non_critical)
    final_bg = len(background_imgs)
    final_total = final_critical + final_non_critical + final_bg

    print(
        f"Split '{split}': {total} imgs (C:{critical_count}, NC:{non_critical_count}, BG:{bg_count})"
    )
    print(
        f"  -> {final_total} imgs (C:{final_critical}, NC:{final_non_critical}, BG:{final_bg})"
    )

    # Copy all Critical, selected Non-Critical, and Background
    for img in critical_imgs:
        copy_image_and_label(img, src_lbl_dir, dst_img_dir, dst_lbl_dir)

    for img in selected_non_critical:
        copy_image_and_label(img, src_lbl_dir, dst_img_dir, dst_lbl_dir)

    for img in background_imgs:
        copy_image_and_label(img, src_lbl_dir, dst_img_dir, dst_lbl_dir)


if __name__ == "__main__":
    print("Balanceando classes: removendo Non-Critical até igualar Critical...")
    DST_ROOT.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        balance_split(split)
    # copiar classes.txt se existir
    classes_txt = SRC_ROOT / "classes.txt"
    if classes_txt.exists():
        shutil.copy2(classes_txt, DST_ROOT / "classes.txt")
    print(f"\nPronto. Dataset balanceado em {DST_ROOT}")
