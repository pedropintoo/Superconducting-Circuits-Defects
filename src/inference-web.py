import io
import zipfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "chip_defect_detection"
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _parse_run_index(run_name: str) -> int:
    suffix = run_name.replace("run", "", 1)
    if suffix == "":
        return 0
    return int(suffix) if suffix.isdigit() else -1


@st.cache_data(show_spinner=False)
def list_available_runs():
    runs = []
    if not RUNS_DIR.exists():
        return runs

    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir() or not run_dir.name.startswith("run"):
            continue

        weight_path = run_dir / "weights" / "best.pt"
        if not weight_path.exists():
            continue

        run_index = _parse_run_index(run_dir.name)
        if run_index < 0:
            continue

        runs.append({
            "label": run_dir.name,
            "index": run_index,
            "weight": weight_path
        })

    return sorted(runs, key=lambda r: r["index"])


@st.cache_resource(show_spinner=False)
def load_model(weight_path: str):
    return YOLO(weight_path)


def _image_from_bytes(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def _annotate_image(model: YOLO, data: bytes, conf: float):
    image = _image_from_bytes(data)
    results = model.predict(image, conf=conf, verbose=False)
    plotted = results[0].plot()
    annotated = Image.fromarray(plotted[:, :, ::-1])  # Convert BGR (Ultralytics) to RGB
    return annotated, results[0]


def _extract_images_from_zip(zip_bytes: bytes):
    images = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue

            suffix = Path(info.filename).suffix.lower()
            if suffix not in SUPPORTED_EXTS:
                continue

            with archive.open(info) as file:
                images.append((Path(info.filename).name, file.read()))

    return images


st.set_page_config(page_title="Chip Defect Detection", layout="wide")
st.title("Chip Defect Detection Demo")
st.caption("Upload a single image or a .zip folder of images to preview YOLO predictions with bounding boxes.")

runs = list_available_runs()
if not runs:
    st.error("No trained runs found under chip_defect_detection. Train a model first to enable the demo.")
    st.stop()

run_labels = [run["label"] for run in runs]
selected_run = st.sidebar.selectbox("Model checkpoint", run_labels, index=len(run_labels) - 1)
selected_weight = next(run for run in runs if run["label"] == selected_run)["weight"]

confidence = st.sidebar.slider("Confidence threshold", min_value=0.05, max_value=0.95, value=0.25, step=0.05)
max_images = st.sidebar.slider("Max images to process", min_value=1, max_value=25, value=10)

model = load_model(str(selected_weight))

input_mode = st.radio(
    "What would you like to upload?",
    ("Single images", "Folder (.zip)"),
    help="Select single images (you can pick multiple files) or upload a zipped folder."
)

pending_images = []

if input_mode == "Single images":
    uploads = st.file_uploader(
        "Select one or more images",
        type=[ext.strip(".") for ext in SUPPORTED_EXTS],
        accept_multiple_files=True
    )
    if uploads:
        for file in uploads:
            file.seek(0)
            data = file.read()
            ext = Path(file.name).suffix.lower()
            if ext not in SUPPORTED_EXTS:
                st.warning(f"Skipping unsupported file: {file.name}")
                continue
            pending_images.append((file.name, data))
else:
    uploaded_zip = st.file_uploader("Upload a .zip archive", type=["zip"])
    if uploaded_zip is not None:
        uploaded_zip.seek(0)
        pending_images = _extract_images_from_zip(uploaded_zip.read())
        if not pending_images:
            st.warning("No supported image files were found inside the archive.")

if pending_images:
    pending_images = pending_images[:max_images]
    st.info(f"Ready to process {len(pending_images)} image(s).")

run_inference = st.button("Generate predictions", disabled=not pending_images)

if run_inference and pending_images:
    with st.spinner("Running inference..."):
        for name, data in pending_images:
            try:
                annotated, result = _annotate_image(model, data, confidence)
            except Exception as exc:  # noqa: BLE001 - present errors to the user
                st.error(f"Failed to process {name}: {exc}")
                continue

            cols = st.columns([1, 1])
            with cols[0]:
                st.image(_image_from_bytes(data), caption=f"Original · {name}", use_container_width=True)
            with cols[1]:
                detections = len(result.boxes) if result.boxes is not None else 0
                st.image(annotated, caption=f"Predictions ({detections} boxes) · {name}", use_container_width=True)

    st.success("Prediction completed.")
    