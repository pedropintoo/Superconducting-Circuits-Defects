"""
Balance classes STRICTLY (Hybrid Approach).
1. Upsamples Criticals to match the original Non-Critical count.
2. Downsamples "Pure" Non-Critical images to remove the excess created by step 1.
Result: Critical and Non-Critical instance counts will be equal.
"""
import random
import shutil
from pathlib import Path

RNG_SEED = 42
ROOT = Path(__file__).resolve().parent
SRC_ROOT = ROOT / "train_val_dataset_sliced"
DST_ROOT = ROOT / "train_val_dataset_sliced_balanced_upsampled"
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

def copy_image_and_label(src_img: Path, src_lbl_dir: Path, dst_img_dir: Path, dst_lbl_dir: Path, new_stem: str = None):
    if new_stem is None:
        new_stem = src_img.stem
    dst_img_name = f"{new_stem}{src_img.suffix}"
    dst_lbl_name = f"{new_stem}.txt"
    shutil.copy2(src_img, dst_img_dir / dst_img_name)
    src_lbl = src_lbl_dir / f"{src_img.stem}.txt"
    dst_lbl = dst_lbl_dir / dst_lbl_name
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
    pure_nc_imgs_data = []  # (path, c_count, nc_count) -> c_count is always 0 here
    background_imgs = []

    total_c = 0
    total_nc = 0

    # 1. Scan and Sort
    for img in images:
        lbl = src_lbl_dir / f"{img.stem}.txt"
        c, nc = get_class_counts(lbl)
        
        total_c += c
        total_nc += nc

        if c > 0:
            critical_imgs_data.append((img, c, nc))
        elif nc > 0:
            pure_nc_imgs_data.append((img, c, nc))
        else:
            background_imgs.append(img)

    print(f"[{split}] Initial: Crit={total_c} | Non-Crit={total_nc}")

    # 2. PHASE 1: UPSAMPLE CRITICALS
    # Target is the current Max (usually Non-Crit)
    target = max(total_c, total_nc)
    needed_c = target - total_c
    
    extras_c = []
    
    if needed_c > 0 and critical_imgs_data:
        print(f"  -> Upsampling Criticals: Need {needed_c} more instances.")
        while needed_c > 0:
            choice = random.choice(critical_imgs_data) # (path, c, nc)
            extras_c.append(choice)
            needed_c -= choice[1]
            # Tracking the side effect:
            total_nc += choice[2]
            total_c += choice[1]

    print(f"  -> Post-Upsample: Crit={total_c} | Non-Crit={total_nc} (Non-Crit grew due to overlap)")

    # 3. PHASE 2: DOWNSAMPLE NON-CRITICALS
    # Now Critical is likely ~6088, but Non-Critical might be ~7300.
    # We remove "Pure Non-Critical" images to bring Non-Critical down to Critical level.
    
    excess_nc = total_nc - total_c
    
    imgs_to_keep_nc = list(pure_nc_imgs_data) # Make a copy
    
    if excess_nc > 0:
        print(f"  -> Downsampling Non-Crit: Removing excess {excess_nc} instances.")
        random.shuffle(imgs_to_keep_nc)
        
        # We need to REMOVE items. 
        
        kept_pure_nc = []
        removed_count = 0
        
        # We want to DROP 'excess_nc'.
        
        # Filter the list
        final_pure_nc = []
        current_excess = excess_nc
        
        for item in imgs_to_keep_nc:
            path, c, nc = item
            if current_excess > 0:
                # remove this image (don't add to final list)
                current_excess -= nc
                removed_count += nc
            else:
                final_pure_nc.append(item)
                
        imgs_to_keep_nc = final_pure_nc
        total_nc -= removed_count

    print(f"  -> Final Balance: Crit={total_c} | Non-Crit={total_nc}")

    # 4. EXECUTE COPY
    # A. Criticals (Originals)
    for img, _, _ in critical_imgs_data:
        copy_image_and_label(img, src_lbl_dir, dst_img_dir, dst_lbl_dir)
        
    # B. Criticals (Upsampled Extras)
    for i, (img, _, _) in enumerate(extras_c):
        new_stem = f"{img.stem}_copy_{i}"
        copy_image_and_label(img, src_lbl_dir, dst_img_dir, dst_lbl_dir, new_stem=new_stem)

    # C. Non-Criticals (Filtered Pure Images)
    for img, _, _ in imgs_to_keep_nc:
        copy_image_and_label(img, src_lbl_dir, dst_img_dir, dst_lbl_dir)

    # D. Backgrounds (All)
    for img in background_imgs:
        copy_image_and_label(img, src_lbl_dir, dst_img_dir, dst_lbl_dir)

if __name__ == "__main__":
    print("Balancing classes STRICT (Hybrid Upsample/Downsample)...")
    DST_ROOT.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        balance_split(split)
        
    classes_txt = SRC_ROOT / "classes.txt"
    if classes_txt.exists():
        shutil.copy2(classes_txt, DST_ROOT / "classes.txt")
    print(f"\nDone. Output: {DST_ROOT}")