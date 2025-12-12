import collections
from pathlib import Path

# --- CONFIGURATION ---
# We check the folder you intend to use as source
DATASET_DIR = Path(__file__).resolve().parent / "new_128_train_val_dataset_sliced_balanced_downsampled_bg8"

def count_dataset():
    if not DATASET_DIR.exists():
        print(f"Error: Folder not found: {DATASET_DIR}")
        return

    # 1. Read Class Names
    classes_file = DATASET_DIR / "classes.txt"
    class_names = {}
    if classes_file.exists():
        print(f"Reading classes from: {classes_file.name}")
        lines = classes_file.read_text(encoding="utf-8").strip().splitlines()
        for idx, name in enumerate(lines):
            class_names[idx] = name
            print(f"  ID {idx}: {name}")
    else:
        print("Warning: classes.txt not found. Showing IDs only.")

    print("-" * 30)

    # 2. Count Instances (TRAIN ONLY)
    total_counts = collections.Counter()
    
    # We only look at 'train' now
    split = "train"
    label_dir = DATASET_DIR / "labels" / split
    
    if label_dir.exists():
        print(f"Scanning split '{split}'...")
        # Get all text files
        label_files = list(label_dir.glob("*.txt"))
        
        for lf in label_files:
            lines = lf.read_text(encoding="utf-8").strip().splitlines()
            for line in lines:
                parts = line.split()
                if not parts: continue
                try:
                    c_id = int(parts[0])
                    total_counts[c_id] += 1
                except ValueError:
                    pass
    else:
        print(f"Error: No labels found in {label_dir}")

    print("-" * 30)
    print("TOTAL INSTANCE COUNTS (TRAIN ONLY):")
    sorted_ids = sorted(total_counts.keys())
    
    for c_id in sorted_ids:
        name = class_names.get(c_id, "Unknown")
        count = total_counts[c_id]
        print(f"  Class ID {c_id} ({name}): {count} instances")

if __name__ == "__main__":
    count_dataset()