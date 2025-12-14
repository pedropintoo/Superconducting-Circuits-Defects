import os
from ultralytics import YOLO
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel
import cv2
import numpy as np

# ----------------------------------------------------------------
# Define which run to use for inference
best = "models/best_model/weights/best.pt"
EXAMPLE = "chips_1"
# ----------------------------------------------------------------

print(f"Using model: {best} for inference.")

examples = {
    "dark_big_burn": [f"../datasets/full_dataset/RQ3_TWPA_V2_W2-251023_Junctions-dark-{i:06d}.jpg" for i in range(280, 290)],
    "dark_open_circuit": [f"../datasets/full_dataset/RQ3_TWPA_V2_W2-251023_Junctions-dark-{i:06d}.jpg" for i in range(240, 250)],
    "dark_burn": [f"../datasets/full_dataset/RQ3_TWPA_V2_W2-251023_Junctions-dark-{i:06d}.jpg" for i in range(1199, 1211)],
    "chips_1": [f"../datasets/full_dataset/RQ3_TWPA_V2_W2-251101_Chips-v2_7500_500_DF-{i:06d}.jpg" for i in range(50, 60)],
    "dark_2": [f"../datasets/full_dataset/RQ3_TWPA_V2_W2-251023_Junctions-dark-{i:06d}.jpg" for i in range(469, 500)],
    "white_chips": [f"../datasets/full_dataset/RQ3_TWPA_V2_W2-251101_Chips-v2_7500_500-{i:06d}.jpg" for i in range(15, 30)],
    "spir_60": [f"../datasets/full_dataset/Second_Batch-PM251015p1-251028_Spir_60_6-bright-{i:06d}.jpg" for i in range(220, 230)],
    "LO_mark": [f"../datasets/full_dataset/Second_Batch-PM251015p1-251022_post_LO_mark-dark-{i:06d}.jpg" for i in range(366, 370)],
    "Val_examples": [f"../datasets/full_dataset/RQ3_TWPA_V2_W2-251023_Junctions-dark-00015{i:01d}.jpg" for i in range(9)],
    "random" : ["../datasets/full_dataset/RQ3_TWPA_V2_W2-251023_Junctions-dark-001274.jpg"]
}

model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=best,
    confidence_threshold=0.5,
    # device="cpu"
)

results = []
for image_path in examples[EXAMPLE]:  # Process all examples
    t0 = cv2.getTickCount()
    result = get_sliced_prediction(
        image_path,
        model,
        slice_height=256,
        slice_width=256,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
    )
    t1 = cv2.getTickCount()
    time_taken = (t1 - t0) / cv2.getTickFrequency()
    print(f"Time taken for sliced inference on {image_path}: {time_taken:.3f} seconds")
    results.append((image_path, result))

    result.export_visuals(export_dir="demo_data/")
    # move to a unique name
    os.rename(
        "demo_data/prediction_visual.png",
        f"demo_data/prediction_visual_{os.path.basename(image_path)}.png",
    )
    print(f"Processed and saved results for image: {image_path}")







## ---------------------------- IMAGE VISUALIZATION ---------------------------- ##

# CV2 to each predict
for image_path, result in results:
    result_image = cv2.imread(f"demo_data/prediction_visual_{os.path.basename(image_path)}.png")
    orig_image = cv2.imread(image_path)

    if result_image is None or orig_image is None:
        print(f"Warning: could not read result and/or original image for {image_path}, skipping display.")
    else:
        def _pad_to_height(img, height, color=255):
            if img.shape[0] == height:
                return img
            pad_total = height - img.shape[0]
            pad_top = pad_total // 2
            pad_bottom = pad_total - pad_top
            return cv2.copyMakeBorder(img, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=[color]*3)

        def _draw_boxes_with_labels(img, preds):
            overlay = img.copy()
            margin = 6  # extra padding around boxes
            thickness = 3
            colors = {
                "Critical": (0, 0, 255),      # red
                "Dirt-Wire": (182, 119, 0),   # ocean blue (BGR)
            }

            h, w = img.shape[:2]
            for obj in preds.object_prediction_list:
                x1, y1, x2, y2 = map(int, obj.bbox.to_xyxy())
                color = colors.get(obj.category.name, (0, 0, 255))
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(w - 1, x2 + margin)
                y2 = min(h - 1, y2 + margin)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)

            return overlay

        def _add_legend(img):
            legend = img.copy()
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.1
            thickness = 2
            entries = [
                ("Critical", (0, 0, 255)),          # red
                ("Dirt-Wire", (182, 119, 0)),       # ocean blue (BGR for #0077b6)
            ]

            box_size = 26
            spacing = 16
            line_gap = 20
            pad = 14

            total_height = len(entries) * (box_size + line_gap) - line_gap + pad * 2
            bg_height = total_height
            bg_width = 280

            h, _ = legend.shape[:2]
            bg_left = 16
            bg_top = max(12, h - bg_height - 16)
            bg_right = bg_left + bg_width
            bg_bottom = bg_top + bg_height

            overlay = legend.copy()
            cv2.rectangle(overlay, (bg_left, bg_top), (bg_right, bg_bottom), (0, 0, 0), -1)
            alpha = 0.45
            legend = cv2.addWeighted(overlay, alpha, legend, 1 - alpha, 0)

            x = bg_left + pad
            y = bg_top + pad + box_size
            for label, color in entries:
                cv2.rectangle(legend, (x, y - box_size), (x + box_size, y), color, -1)
                cv2.putText(legend, label, (x + box_size + spacing, y - 6), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
                y += box_size + line_gap

            return legend

        max_height = orig_image.shape[0]

        labeled_orig = _draw_boxes_with_labels(orig_image, result)
        labeled_orig = _add_legend(labeled_orig)

        # Titles with semi-transparent backgrounds for readability
        def _put_title(img, text, pos, color):
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.3
            thickness = 3
            (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            x, y = pos
            pad = 10
            bg_x1, bg_y1 = x - pad, y - th - pad
            bg_x2, bg_y2 = x + tw + pad, y + baseline + pad
            overlay = img.copy()
            cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
            img[:] = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)
            cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

        _put_title(labeled_orig, "Prediction", (24, 64), (0, 220, 0))

        padded_orig = _pad_to_height(labeled_orig, max_height)

        side_by_side = padded_orig

        win_name = f"Prediction · {os.path.basename(image_path)}"
        try:
            cv2.startWindowThread()
        except Exception:
            pass
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.imshow(win_name, side_by_side)

        while True:
            key = cv2.waitKey(100) & 0xFF
            if key == 27 or key == ord("q"):
                break
            try:
                if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except Exception:
                break

        cv2.destroyWindow(win_name)
        cv2.waitKey(1)
        cv2.destroyAllWindows()
