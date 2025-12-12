"""
Balance classes STRICTLY (Downsampling Approach).
1. Counts Total Critical Instances (Target).
2. Counts Non-Critical instances already present in Critical images (Overlap).
3. Selects just enough "Pure" Non-Critical images to bridge the gap.
Result: Critical and Non-Critical instance counts will be equal (unless Critical images alone already contain excess Non-Criticals).
- Source: `train_val_dataset_sliced`
- Output: `train_val_dataset_sliced_balanced_downsampled`
"""
import random
import shutil
from pathlib import Path

RNG_SEED = 42
ROOT = Path(__file__).resolve().parent
SRC_ROOT = ROOT / "new_128_train_val_dataset_sliced"
DST_ROOT = ROOT / "new_128_train_val_dataset_sliced_balanced_downsampled"
SPLITS = ["train", "val"]
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# CONFIG
CRITICAL_CLASS = 0
NON_CRITICAL_CLASS = 1

random.seed(RNG_SEED)

def list_images(images_dir: Path):
    return sorted([p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS])

def get_class_counts(label_path: Path):
    c_count = 0
    nc_count = 0
    if not label_path.exists():
        return 0, 0
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return 0, 0 
    for line in text.split("\n"):
        parts = line.strip().split()
        if len(parts) >= 1:
            try:
                cls = int(parts[0])
                if cls == CRITICAL_CLASS:
                    c_count += 1
                elif cls == NON_CRITICAL_CLASS:
                    nc_count += 1
            except ValueError:
                pass
    return c_count, nc_count

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def copy_image_and_label(src_img: Path, src_lbl_dir: Path, dst_img_dir: Path, dst_lbl_dir: Path):
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
        return

    # Data Structures
    critical_imgs_data = [] # (path, c_count, nc_count)
    pure_nc_imgs_data = []  # (path, c_count, nc_count) -> c_count is always 0
    background_imgs = []

    total_c = 0
    total_nc_overlap = 0 # Non-criticals found inside critical images

    # 1. Scan and Categorize
    for img in images:
        lbl = src_lbl_dir / f"{img.stem}.txt"
        c, nc = get_class_counts(lbl)
        
        if c > 0:
            critical_imgs_data.append((img, c, nc))
            total_c += c
            total_nc_overlap += nc
        elif nc > 0:
            pure_nc_imgs_data.append((img, c, nc))
        else:
            background_imgs.append(img)

    print(f"[{split}] Critical Files: {len(critical_imgs_data)} | Pure Non-Crit Files: {len(pure_nc_imgs_data)}")
    print(f"[{split}] Critical Instances: {total_c} | Overlap Non-Crit Instances: {total_nc_overlap}")

    # 2. CALCULATE BUDGET
    # We want Total Non-Crit == Total Critical
    # So: (Overlap NC) + (Selected Pure NC) = Total Critical
    needed_nc = total_c - total_nc_overlap
    
    selected_pure_nc = []
    
    if needed_nc <= 0:
        print(f"  [WARN] Critical images already contain {total_nc_overlap} Non-Criticals (Target: {total_c}).")
        print("  -> Impossible to balance perfectly by downsampling alone. Keeping 0 pure Non-Criticals.")
        final_nc_count = total_nc_overlap
    else:
        print(f"  -> Need {needed_nc} more Non-Critical instances from pure images.")
        random.shuffle(pure_nc_imgs_data)
        
        current_added = 0
        for item in pure_nc_imgs_data:
            path, _, nc = item
            if current_added + nc <= needed_nc:
                selected_pure_nc.append(item)
                current_added += nc
            
            if current_added >= needed_nc:
                break
        
        final_nc_count = total_nc_overlap + current_added
        print(f"  -> Selected {len(selected_pure_nc)} pure images providing {current_added} instances.")

    print(f"  -> Final Balance: Crit={total_c} | Non-Crit={final_nc_count}")

    # 3. EXECUTE COPY
    # A. Criticals (All)
    for img, _, _ in critical_imgs_data:
        copy_image_and_label(img, src_lbl_dir, dst_img_dir, dst_lbl_dir)
        
    # B. Non-Criticals (Selected Pure)
    for img, _, _ in selected_pure_nc:
        copy_image_and_label(img, src_lbl_dir, dst_img_dir, dst_lbl_dir)

    # C. Backgrounds (All - passed to next stage)
    for img in background_imgs:
        copy_image_and_label(img, src_lbl_dir, dst_img_dir, dst_lbl_dir)

if __name__ == "__main__":
    print("Balancing classes STRICT (Downsampling Non-Criticals)...")
    DST_ROOT.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        balance_split(split)
        
    classes_txt = SRC_ROOT / "classes.txt"
    if classes_txt.exists():
        shutil.copy2(classes_txt, DST_ROOT / "classes.txt")
    print(f"\nDone. Output: {DST_ROOT}")