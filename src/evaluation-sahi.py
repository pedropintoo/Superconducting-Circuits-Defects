"""\
Inference script using SAHI for sliced inference with YOLO models on full images.

Authors: Pedro Pinto, João Pinto, Fedor Chikhachev
"""
import os
import argparse
import json
import glob
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
import re

# SAHI imports
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# COCO Evaluation imports
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# --- CONFIGURATION ---
CLASS_NAMES = {0: "Critical", 1: "Dirt-Wire"}
SLICE_HEIGHT = 256
SLICE_WIDTH = 256
OVERLAP_HEIGHT_RATIO = 0.2
OVERLAP_WIDTH_RATIO = 0.2
CONFIDENCE_THRESHOLD = 0.5  # Fixed
IOU_THRESHOLD = 0.5  # For Confusion Matrix matching

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO + SAHI Inference and Evaluation on Full Images")
    parser.add_argument("--split", type=str, default="val", help="Dataset split to evaluate")
    parser.add_argument("--weights", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--data_root", type=str, default="../datasets/new_train_val_dataset", help="Root path of dataset")
    parser.add_argument("--project", type=str, default="inference_results_sahi", help="Save results to project/name")
    parser.add_argument("--name", type=str, default="exp", help="Save results to project/name")
    return parser.parse_args()

def increment_path(path, exist_ok=False, sep='', mkdir=False):
    """
    Increment file or directory path, i.e. runs/exp -> runs/exp{sep}2, runs/exp{sep}3, ...
    """
    path = Path(path)
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        dirs = glob.glob(f"{path}{sep}*")
        matches = [re.search(rf"{path.name}{sep}(\d+)", d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        path = Path(f"{path}{sep}{n}{suffix}")
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)
    return path

def xywhn2xywh(x, w, h):
    """Convert normalized xywh (YOLO) to pixel xywh (COCO)."""
    bbox_width = x[3] * w
    bbox_height = x[4] * h
    x_center = x[1] * w
    y_center = x[2] * h
    x_min = x_center - (bbox_width / 2)
    y_min = y_center - (bbox_height / 2)
    return [x[0], x_min, y_min, bbox_width, bbox_height]

def compute_iou(box1, box2):
    """Compute IoU between two boxes [x, y, w, h]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

def draw_boxes(image, gt_boxes, pred_boxes, save_path):
    """
    Draws GT (Green) and Predictions (Red) on the image and saves it.
    gt_boxes: list of [cls_id, x, y, w, h]
    pred_boxes: list of [cls_id, x, y, w, h, score]
    """
    vis_img = image.copy()
    
    # Draw Ground Truth (Green)
    for box in gt_boxes:
        cls_id = int(box[0])
        x, y, w, h = map(int, box[1:])
        label = f"GT: {CLASS_NAMES.get(cls_id, cls_id)}"
        
        # Draw Rectangle
        cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Draw Label Background
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis_img, (x, y - 20), (x + text_w, y), (0, 255, 0), -1)
        # Draw Text
        cv2.putText(vis_img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Draw Predictions (Red)
    for box in pred_boxes:
        cls_id = int(box[0])
        x, y, w, h = map(int, box[1:5])
        score = box[5]
        label = f"{CLASS_NAMES.get(cls_id, cls_id)} {score:.2f}"
        
        # Draw Rectangle
        cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # Draw Label Background
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # Offset prediction label slightly if it overlaps GT exactly, or put it at bottom
        cv2.rectangle(vis_img, (x, y + h), (x + text_w, y + h + 20), (0, 0, 255), -1)
        # Draw Text
        cv2.putText(vis_img, label, (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imwrite(str(save_path), vis_img)

class ConfusionMatrix:
    def __init__(self, num_classes, labels):
        # Internal: Row=True, Col=Predicted
        self.matrix = np.zeros((num_classes + 1, num_classes + 1))
        self.num_classes = num_classes
        self.labels = labels

    def process_batch(self, detections, labels):
        """
        detections: list of [cls, x, y, w, h, score] (score is ignored here but passed)
        labels: list of [cls, x, y, w, h]
        """
        detection_matched = [False] * len(detections)
        for label in labels:
            gt_cls = int(label[0])
            best_iou = 0
            best_det_idx = -1
            
            for i, det in enumerate(detections):
                if detection_matched[i]: continue
                # det[1:5] is x,y,w,h
                iou = compute_iou(label[1:], det[1:5])
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = i
            
            if best_iou > IOU_THRESHOLD:
                if best_det_idx >= 0:
                    pred_cls = int(detections[best_det_idx][0])
                    self.matrix[gt_cls, pred_cls] += 1
                    detection_matched[best_det_idx] = True
                else:
                    self.matrix[gt_cls, self.num_classes] += 1 # FN
            else:
                self.matrix[gt_cls, self.num_classes] += 1 # FN

        for i, matched in enumerate(detection_matched):
            if not matched:
                pred_cls = int(detections[i][0])
                self.matrix[self.num_classes, pred_cls] += 1 # FP

    def get_metrics(self):
        """Calculate Precision, Recall, and F1-Score per class."""
        metrics = {}
        f1_scores = []
        
        for i in range(self.num_classes):
            tp = self.matrix[i, i]
            fp = self.matrix[:, i].sum() - tp
            fn = self.matrix[i, :].sum() - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            class_name = self.labels.get(i, f"Class {i}")
            metrics[class_name] = {
                "Precision": precision,
                "Recall": recall,
                "F1": f1
            }
            f1_scores.append(f1)
            
        metrics["Mean F1"] = np.mean(f1_scores) if f1_scores else 0
        return metrics

    def plot(self, save_dir="."):
        plot_matrix = self.matrix.T
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        labels_ext = [self.labels.get(i, f"Class {i}") for i in range(self.num_classes)] + ["background"]
        
        sns.heatmap(plot_matrix, annot=True, cmap="Blues", fmt=".0f", 
                    xticklabels=labels_ext, yticklabels=labels_ext)
        
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title(f"Confusion Matrix (Conf={CONFIDENCE_THRESHOLD})")
        
        save_path = os.path.join(save_dir, "confusion_matrix_sahi.png")
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
        plt.close()

def main():
    args = parse_args()
    
    # 1. Setup Output Directory
    save_dir = increment_path(Path(args.project) / args.name, exist_ok=False)
    save_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir = save_dir / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nResults will be saved to: {save_dir}")

    # 2. Setup Paths
    images_path = os.path.join(args.data_root, "images", args.split)
    labels_path = os.path.join(args.data_root, "labels", args.split)
    
    if not os.path.exists(images_path):
        raise FileNotFoundError(f"Images folder not found: {images_path}")

    # 3. Initialize Model
    print(f"Loading model: {args.weights}")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=args.weights,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # 4. Prepare COCO structures
    coco_gt = {
        "info": {"description": "SAHI Eval"},
        "images": [],
        "annotations": [],
        "categories": [{"id": k, "name": v} for k, v in CLASS_NAMES.items()]
    }
    coco_dt = []
    
    confusion_matrix = ConfusionMatrix(len(CLASS_NAMES), CLASS_NAMES)
    
    image_files = sorted(glob.glob(os.path.join(images_path, "*")))
    image_files = [f for f in image_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    print(f"Found {len(image_files)} images. Running Inference (Conf={CONFIDENCE_THRESHOLD})...")

    ann_id_counter = 1
    
    # 5. Inference Loop
    for img_idx, img_path in tqdm(enumerate(image_files), total=len(image_files), desc="Inference"):
        filename = os.path.basename(img_path)
        label_file = os.path.join(labels_path, os.path.splitext(filename)[0] + ".txt")
        
        image = cv2.imread(img_path)
        if image is None: continue
        h, w, _ = image.shape
        
        # --- A. Process Ground Truth ---
        coco_gt["images"].append({"id": img_idx, "width": w, "height": h, "file_name": filename})
        gt_boxes_vis = [] # [cls, x, y, w, h]
        
        if os.path.exists(label_file):
            with open(label_file, "r") as f:
                for line in f:
                    parts = list(map(float, line.strip().split()))
                    cls_id = int(parts[0])
                    abs_box = xywhn2xywh(parts, w, h)
                    
                    coco_gt["annotations"].append({
                        "id": ann_id_counter,
                        "image_id": img_idx,
                        "category_id": cls_id,
                        "bbox": abs_box[1:],
                        "area": abs_box[3] * abs_box[4],
                        "iscrowd": 0
                    })
                    ann_id_counter += 1
                    gt_boxes_vis.append([cls_id] + abs_box[1:])

        # --- B. Run SAHI Inference ---
        result = get_sliced_prediction(
            img_path,
            detection_model,
            slice_height=SLICE_HEIGHT,
            slice_width=SLICE_WIDTH,
            overlap_height_ratio=OVERLAP_HEIGHT_RATIO,
            overlap_width_ratio=OVERLAP_WIDTH_RATIO,
            verbose=0
        )

        # --- C. Process Predictions ---
        pred_boxes_vis = [] # [cls, x, y, w, h, score]
        for obj in result.object_prediction_list:
            cls_id = obj.category.id
            score = obj.score.value
            
            box_w = obj.bbox.maxx - obj.bbox.minx
            box_h = obj.bbox.maxy - obj.bbox.miny
            coco_box = [obj.bbox.minx, obj.bbox.miny, box_w, box_h]
            
            coco_dt.append({
                "image_id": img_idx,
                "category_id": cls_id,
                "bbox": coco_box,
                "score": score
            })

            # For CM and Vis: [cls, x, y, w, h, score]
            pred_boxes_vis.append([cls_id] + coco_box + [score])

        # --- D. Update Confusion Matrix ---
        confusion_matrix.process_batch(pred_boxes_vis, gt_boxes_vis)
        
        # --- E. VISUALIZATION ---
        draw_boxes(image, gt_boxes_vis, pred_boxes_vis, visuals_dir / filename)

    # 6. Save COCO JSONs
    gt_json_path = os.path.join(save_dir, "coco_gt.json")
    dt_json_path = os.path.join(save_dir, "coco_dt.json")
    
    with open(gt_json_path, "w") as f:
        json.dump(coco_gt, f)
    with open(dt_json_path, "w") as f:
        json.dump(coco_dt, f)

    # 7. Metrics Calculation & Saving
    print("\n--- Calculating Metrics ---")
    
    cm_metrics = confusion_matrix.get_metrics()
    
    try:
        coco_gt_obj = COCO(gt_json_path)
        coco_dt_obj = coco_gt_obj.loadRes(dt_json_path)
        coco_eval = COCOeval(coco_gt_obj, coco_dt_obj, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        map50 = coco_eval.stats[1]
        map50_95 = coco_eval.stats[0]
    except Exception as e:
        print(f"Error during COCO evaluation: {e}")
        map50 = 0.0
        map50_95 = 0.0

    # Display Metrics to Console
    print("\n" + "="*40)
    print(f"  RESULTS (Conf={CONFIDENCE_THRESHOLD})")
    print("="*40)
    print(f"{'Class':<15} | {'Prec.':<8} | {'Recall':<8} | {'F1':<8}")
    print("-" * 45)
    for cls_name, vals in cm_metrics.items():
        if cls_name == "Mean F1": continue
        print(f"{cls_name:<15} | {vals['Precision']:.4f}   | {vals['Recall']:.4f}   | {vals['F1']:.4f}")
    print("-" * 45)
    print(f"MEAN F1 SCORE   : {cm_metrics['Mean F1']:.4f}")
    print(f"mAP @ 0.50      : {map50:.4f}")
    print("="*40 + "\n")

    # Save to file
    with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
        f.write(f"Model: {args.weights}\n")
        f.write(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}\n")
        f.write(f"Evaluation Split: {args.split}\n\n")
        
        f.write("=== COCO Metrics ===\n")
        f.write(f"mAP @ 0.50: {map50:.4f}\n")
        f.write(f"mAP @ 0.50:0.95: {map50_95:.4f}\n\n")
        
        f.write("=== Class-wise Performance (from Confusion Matrix) ===\n")
        for cls_name, vals in cm_metrics.items():
            if cls_name == "Mean F1": continue
            f.write(f"Class: {cls_name}\n")
            f.write(f"  Precision: {vals['Precision']:.4f}\n")
            f.write(f"  Recall:    {vals['Recall']:.4f}\n")
            f.write(f"  F1-Score:  {vals['F1']:.4f}\n")
            f.write("-" * 20 + "\n")
            
        f.write(f"\nMean F1 Score: {cm_metrics['Mean F1']:.4f}\n")

    # 8. Plot Confusion Matrix
    print("Generating Confusion Matrix...")
    confusion_matrix.plot(save_dir=save_dir)

if __name__ == "__main__":
    main()