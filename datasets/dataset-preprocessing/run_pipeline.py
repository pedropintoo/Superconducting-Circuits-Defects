"""Batch preprocessing runner for YOLO/COCO datasets.

This script chains optional steps: prepare labels and split, YOLO->COCO, slice COCO, COCO->YOLO, class balancing, and background balancing.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List

from actions import (
    balance_backgrounds,
    balance_classes_downsample,
    balance_classes_upsample,
    coco_to_yolo,
    prepare_labels_and_split,
    slice_coco,
    yolo_to_coco,
)

rng_seed = 42

ACTIONS_ALL = [
    "prepare_labels_and_split",
    "yolo_to_coco",
    "slice_coco",
    "coco_to_yolo",
    "balance_classes_upsample",
    "balance_backgrounds",
]

ACTION_DESCRIPTIONS = {
    "prepare_labels_and_split": "Prepare labels-only set, map images, split",
    "yolo_to_coco": "Export YOLO images/labels to COCO json",
    "slice_coco": "Tile COCO images/annotations with overlap",
    "coco_to_yolo": "Convert COCO slices back to YOLO",
    "balance_backgrounds": "Downsample/keep backgrounds to target ratio",
    "balance_classes_downsample": "Downsample Dirt-Wire to Critical count",
    "balance_classes_upsample": "Hybrid up/down sample Dirt-Wire vs Critical",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset preprocessing pipeline")
    parser.add_argument("--actions", nargs="+", default=ACTIONS_ALL, help="Sequence of actions to run")
    parser.add_argument("--input-dir", type=Path, default=Path("train_val_dataset"))
    parser.add_argument("--labels-root", type=Path, default=Path("train_val_dataset"), help="Directory containing YOLO labels (or labels/ subdir)")
    parser.add_argument("--images-root", type=Path, default=Path("full_dataset/images"), help="Directory containing full images to map labels")
    parser.add_argument("--classes", type=dict, default={0: "Critical", 1: "Dirt-Wire"})
    parser.add_argument("--background-target", type=float, default=0.08)
    parser.add_argument("--slice-height", type=int, default=128)
    parser.add_argument("--slice-width", type=int, default=128)
    parser.add_argument("--slice-overlap", type=float, default=0.2)
    parser.add_argument("--split-ratios", nargs="+", type=float, default=[0.8, 0.2], help="Ratios for prepare_labels_and_split (len 2=train/val or 3=train/val/test)")
    parser.add_argument("--clean-targets", action="store_true", help="Clean intermediate output folders before running each step")
    parser.add_argument("--output-dir", type=Path, default=Path("../train_val_dataset_preprocessed"))
    parser.add_argument("--splits-output-dir", type=Path, default=Path("../train_val_dataset"), help="YOLO format output directory for prepare_labels_and_split")
    return parser.parse_args()


def maybe_clean(path: Path, enabled: bool) -> None:
    if enabled and path.exists():
        shutil.rmtree(path)


def run_pipeline(args: argparse.Namespace) -> None:
    actions: List[str] = args.actions
    # Derive splits from split-ratios
    if len(args.split_ratios) == 2:
        splits = ["train", "val"]
    elif len(args.split_ratios) == 3:
        splits = ["train", "val", "test"]
    else:
        raise ValueError("split_ratios must have length 2 or 3")
    classes = args.classes

    # Ensure background balancing happens after any class balancing steps
    if "balance_backgrounds" in actions and ("balance_classes_downsample" in actions or "balance_classes_upsample" in actions):
        actions = [a for a in actions if a != "balance_backgrounds"] + ["balance_backgrounds"]

    current_yolo = args.input_dir
    temp_root = Path("tmp_pipeline")
    split_out = temp_root / "split_yolo"
    coco_out = temp_root / "coco"
    sliced_out = temp_root / "coco_sliced"
    yolo_from_coco = temp_root / "yolo_from_coco"
    bg_out = temp_root / "balance_backgrounds"
    class_out = temp_root / "class_balance"

    # Always start clean for deterministic outputs
    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir(parents=True, exist_ok=True)
    current_coco = coco_out

    print("\nPipeline actions to run: " + " -> ".join(actions) + "\n")

    config_rows = [
        ("splits", ", ".join(splits)),
        ("input_dir", str(args.input_dir)),
        ("labels_root", str(args.labels_root)),
        ("images_root", str(args.images_root)),
        ("output_dir", str(args.output_dir)),
        ("coco_out", str(coco_out)),
        ("sliced_out", str(sliced_out)),
        ("yolo_from_coco", str(yolo_from_coco)),
        ("bg_out", str(bg_out)),
        ("class_out", str(class_out)),
        ("slice", f"{args.slice_height}x{args.slice_width}, overlap {args.slice_overlap}"),
        ("background_target", f"{args.background_target:.3f}"),
        ("split_ratios", ",".join(map(str, args.split_ratios))),
    ]
    kv_header = f"{'Key':<18} | Value"
    print(kv_header)
    print("-" * len(kv_header))
    for key, val in config_rows:
        print(f"{key:<18} | {val}")
    print()

    for action in actions:
        print(f"[RUN] {action}")
        if action == "prepare_labels_and_split":
            maybe_clean(split_out, True)
            current_yolo = prepare_labels_and_split(
                labels_root=args.labels_root,
                images_root=args.images_root,
                yolo_out=split_out,
                split_ratios=args.split_ratios,
                rng_seed=rng_seed,
                splits_override=None,
            )
            new_dir = args.splits_output_dir
            if new_dir.exists():
                shutil.rmtree(new_dir)
            shutil.copytree(current_yolo, new_dir)
            print(f"\nDone. Output copies to: {new_dir}\n")
            continue

        if action == "yolo_to_coco":
            maybe_clean(coco_out, True)
            current_coco = yolo_to_coco(current_yolo, classes, coco_out, splits)

        elif action == "slice_coco":
            maybe_clean(sliced_out, True)
            source_coco = current_coco
            current_coco = slice_coco(
                source_coco,
                sliced_out,
                slice_height=args.slice_height,
                slice_width=args.slice_width,
                overlap_height_ratio=args.slice_overlap,
                overlap_width_ratio=args.slice_overlap,
                splits=splits,
            )

        elif action == "coco_to_yolo":
            maybe_clean(yolo_from_coco, True)
            current_yolo = coco_to_yolo(current_coco, yolo_from_coco, splits=splits)

        elif action == "balance_backgrounds":
            target = args.background_target
            maybe_clean(bg_out, True)
            current_yolo = balance_backgrounds(current_yolo, bg_out, target_bg_ratio=target, splits=splits, rng_seed=rng_seed)

        elif action in {"balance_classes_downsample", "balance_classes_upsample"}:
            out_dir = class_out
            maybe_clean(out_dir, True)
            if action == "balance_classes_downsample":
                current_yolo = balance_classes_downsample(
                    current_yolo,
                    out_dir,
                    critical_id=0,
                    dirt_wire_id=1,
                    splits=splits,
                    rng_seed=rng_seed,
                )
            elif action == "balance_classes_upsample":
                current_yolo = balance_classes_upsample(
                    current_yolo,
                    out_dir,
                    critical_id=0,
                    dirt_wire_id=1,
                    splits=splits,
                    rng_seed=rng_seed,
                )
        else:
            print(f"[WARN] Unknown action '{action}', skipping")

    final_output = args.output_dir
    if final_output.exists():
        shutil.rmtree(final_output)
    shutil.copytree(current_yolo, final_output)
    print(f"\nFinal dataset copied to: {final_output}")

    if args.clean_targets and temp_root.exists():
        shutil.rmtree(temp_root)

    print(f"\nPipeline finished. Final YOLO dataset at: {current_yolo}")


if __name__ == "__main__":
    run_pipeline(parse_args())
