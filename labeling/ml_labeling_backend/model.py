import os
import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Helper: locate the latest trained YOLO run
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT.parent.parent / "src" / "chip_defect_detection"

# Base path that Label Studio's local-files storage is relative to.
# Adjust if your LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT differs.
LOCAL_FILES_BASE = Path.home()


def _parse_run_index(run_name: str) -> int:
    suffix = run_name.replace("run", "", 1)
    if suffix == "":
        return 0
    return int(suffix) if suffix.isdigit() else -1


def get_latest_model_path() -> Path:
    """Return the path to the best.pt weights from the most recent training run."""
    if not RUNS_DIR.exists():
        raise FileNotFoundError(f"Runs directory not found: {RUNS_DIR}")

    runs = []
    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir() or not run_dir.name.startswith("run"):
            continue
        weight_path = run_dir / "weights" / "best.pt"
        if not weight_path.exists():
            continue
        idx = _parse_run_index(run_dir.name)
        if idx >= 0:
            runs.append((idx, weight_path))

    if not runs:
        raise FileNotFoundError("No trained runs with best.pt found under " + str(RUNS_DIR))

    runs.sort(key=lambda x: x[0])
    return runs[-1][1]


def resolve_local_path(url: str) -> str:
    """
    Convert a Label Studio image URL to an absolute filesystem path.

    Handles:
      • /data/local-files/?d=<relative_path>  →  LOCAL_FILES_BASE / relative_path
      • /data/upload/...                      →  raises (needs LABEL_STUDIO_URL)
      • Absolute filesystem paths             →  returned as-is
    """
    # Already an absolute path on disk
    if url.startswith("/") and not url.startswith("/data/"):
        return url

    parsed = urlparse(url)

    # Local-files storage: /data/local-files/?d=Documents/...
    if parsed.path == "/data/local-files/":
        qs = parse_qs(parsed.query)
        rel = qs.get("d", [None])[0]
        if rel:
            return str(LOCAL_FILES_BASE / rel)

    raise FileNotFoundError(
        f"Cannot resolve image URL to a local path: {url}. "
        "Set LABEL_STUDIO_URL and LABEL_STUDIO_API_KEY if images are stored remotely."
    )


# ---------------------------------------------------------------------------
# Label Studio ML Backend
# ---------------------------------------------------------------------------
CONFIDENCE_FACTOR = 0.2
class NewModel(LabelStudioMLBase):
    """YOLO-based ML Backend for chip defect detection."""

    def setup(self):
        """Load the YOLO model once when the backend starts."""
        model_path = get_latest_model_path()
        self.model = YOLO(str(model_path))
        self.set("model_version", model_path.parent.parent.name)  # e.g. "run5"
        print(f"[ML Backend] Loaded model from {model_path}")

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """
        Run YOLO inference on the images referenced by each task.

        Expected task format (image stored in Label Studio):
            {"data": {"image": "/data/upload/1/abc.jpg"}}

        The method returns predictions in Label Studio's rectanglelabels format.
        """
        from_name = "label"
        to_name = "image"
        # Try to infer from_name/to_name from the parsed label config
        if self.parsed_label_config:
            for key, info in self.parsed_label_config.items():
                if info.get("type") == "RectangleLabels":
                    from_name = key
                    to_name = info.get("to_name", ["image"])[0]
                    break

        predictions = []
        for task in tasks:
            # Download image from Label Studio storage
            image_url = task["data"].get("image")
            if not image_url:
                predictions.append({"result": []})
                continue

            # Resolve local-files URLs directly; fall back to SDK for remote storage
            try:
                local_path = resolve_local_path(image_url)
            except FileNotFoundError:
                local_path = self.get_local_path(image_url, task_id=task.get("id"))

            # Run inference
            results = self.model.predict(local_path, conf=CONFIDENCE_FACTOR, verbose=False)
            result_obj = results[0]

            # Convert YOLO boxes to Label Studio format
            task_results = []
            img_width, img_height = result_obj.orig_shape[1], result_obj.orig_shape[0]

            if result_obj.boxes is not None:
                for box in result_obj.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    label = self.model.names.get(cls_id, str(cls_id))

                    # Label Studio expects percentages
                    x_pct = (x1 / img_width) * 100
                    y_pct = (y1 / img_height) * 100
                    w_pct = ((x2 - x1) / img_width) * 100
                    h_pct = ((y2 - y1) / img_height) * 100

                    task_results.append({
                        "id": str(uuid.uuid4())[:8],
                        "from_name": from_name,
                        "to_name": to_name,
                        "type": "rectanglelabels",
                        "value": {
                            "x": x_pct,
                            "y": y_pct,
                            "width": w_pct,
                            "height": h_pct,
                            "rotation": 0,
                            "rectanglelabels": [label],
                        },
                        "score": conf,
                    })

            predictions.append({
                "model_version": self.get("model_version"),
                "result": task_results,
            })

        return ModelResponse(predictions=predictions)

    def fit(self, event, data, **kwargs):  # noqa: ARG002
        """
        Called on annotation events. Currently a no-op; online training is not implemented.
        """
        print(f"[ML Backend] fit() called with event={event}. No action taken.")

