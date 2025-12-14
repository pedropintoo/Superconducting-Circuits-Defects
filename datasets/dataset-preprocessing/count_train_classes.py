import collections
from pathlib import Path

# --- CONFIGURATION ---
DATASET_DIR = Path(__file__).resolve().parent / "../processed_dataset"
DEFAULT_CLASS_NAMES = {0: "Critical", 1: "Dirt-Wire"}


def read_class_names(classes_file: Path) -> dict[int, str]:
    """Return a mapping of class id to name, falling back to defaults."""
    if not classes_file.exists():
        return DEFAULT_CLASS_NAMES.copy()

    lines = classes_file.read_text(encoding="utf-8").strip().splitlines()
    class_names = {idx: name for idx, name in enumerate(lines)}
    # Ensure the requested labels are always present
    class_names.setdefault(0, DEFAULT_CLASS_NAMES[0])
    class_names.setdefault(1, DEFAULT_CLASS_NAMES[1])
    return class_names


def count_split(label_dir: Path) -> collections.Counter:
    """Count instances per class id under a split directory."""
    counts = collections.Counter()
    if not label_dir.exists():
        return counts

    for lf in label_dir.glob("*.txt"):
        for line in lf.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if not parts:
                continue
            try:
                counts[int(parts[0])] += 1
            except ValueError:
                continue
    return counts


def print_counts(split: str, counts: collections.Counter, class_names: dict[int, str]):
    print("-" * 30)
    print(f"TOTAL INSTANCE COUNTS ({split.upper()}):")
    for c_id in sorted(counts.keys()):
        name = class_names.get(c_id, "Unknown")
        print(f"  Class ID {c_id} ({name}): {counts[c_id]} instances")


def count_dataset():
    if not DATASET_DIR.exists():
        print(f"Error: Folder not found: {DATASET_DIR}")
        return

    classes_file = DATASET_DIR / "classes.txt"
    class_names = read_class_names(classes_file)

    for split in ("train", "val"):
        label_dir = DATASET_DIR / "labels" / split
        counts = count_split(label_dir)
        if not counts:
            print(f"Warning: No labels found in {label_dir}")
        print_counts(split, counts, class_names)


if __name__ == "__main__":
    count_dataset()