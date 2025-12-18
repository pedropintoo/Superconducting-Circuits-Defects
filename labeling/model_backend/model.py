"""\
Model: YOLO-based ML Backend for Label Studio

Requirements:
- Install requirements.txt in a python virtual environment.

Usage: 
    CONFIDENCE_FACTOR=0.5 SLICE_HEIGHT=256 SLICE_WIDTH=256 SLICE_OVERLAP=0.2 \
    label-studio-ml start .

Authors: Pedro Pinto, João Pinto, Fedor Chikhachev
"""

import os
import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from PIL import Image

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper: locate the latest trained YOLO run
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT.parent.parent / "src" / "models"

# Base path that Label Studio's local-files storage is relative to.
# Adjust if your LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT differs.
LOCAL_FILES_BASE = Path.home()


def _parse_run_index(run_name: str) -> int:
    suffix = run_name.replace("run", "", 1)
    if suffix == "":
        return 0
    return int(suffix) if suffix.isdigit() else -1

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
# Allow overrides via environment variables for quick tuning.
# ---------------------------------------------------------------------------
def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


CONFIDENCE_FACTOR = _env_float("CONFIDENCE_FACTOR", 0.5)
SLICE_HEIGHT = _env_int("SLICE_HEIGHT", 256)
SLICE_WIDTH = _env_int("SLICE_WIDTH", 256)
SLICE_OVERLAP = _env_float("SLICE_OVERLAP", 0.2)


class NewModel(LabelStudioMLBase):
    """YOLO-based ML Backend for chip defect detection."""

    def setup(self):
        """Load the YOLO model once when the backend starts."""
        model_path = RUNS_DIR / "best_model" / "weights" / "best.pt"
        self.model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=str(model_path),
            confidence_threshold=CONFIDENCE_FACTOR,
        )
        self.set("model_version", model_path.parent.parent.name)  # e.g. "run5"
        logger.warning(
            f"[ML Backend] Loaded model from {model_path} | conf={CONFIDENCE_FACTOR} "
            f"slice=({SLICE_HEIGHT}x{SLICE_WIDTH}) overlap={SLICE_OVERLAP}"
        )

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

            # Run sliced inference via SAHI
            pil_image = Image.open(local_path).convert("RGB")
            result_obj = get_sliced_prediction(
                pil_image,
                self.model,
                slice_height=SLICE_HEIGHT,
                slice_width=SLICE_WIDTH,
                overlap_height_ratio=SLICE_OVERLAP,
                overlap_width_ratio=SLICE_OVERLAP,
                verbose=0,
            )

            # Convert predictions to Label Studio format
            task_results = []
            img_width, img_height = pil_image.size

            for obj in result_obj.object_prediction_list:
                x1, y1, x2, y2 = obj.bbox.to_xyxy()
                conf = float(obj.score.value)
                label = obj.category.name

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
        logger.warning(f"[ML Backend] fit() called with event={event}. No action taken.")

