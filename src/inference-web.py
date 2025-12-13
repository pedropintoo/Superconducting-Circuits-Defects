"""\
Inference web app for chip defect detection using SAHI-sliced YOLO models.

Authors: Pedro Pinto, João Pinto, Fedor Chikhachev
"""
import base64
import io
import time
import zipfile
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel


ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "models"
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEMO_IMAGE_PATHS = [
    ROOT / ".." / "datasets" / "full_dataset" / f"Second_Batch-PM250715p2-251103_Post_DUV_Strip-dark-{i:06d}.jpg" for i in range(37, 48)
]


@st.cache_data(show_spinner=False)
def list_available_runs():
    runs = []
    if not RUNS_DIR.exists():
        return runs
    run_index = 0
    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue

        weight_path = run_dir / "weights" / "best.pt"
        if not weight_path.exists():
            continue
        
        run_index += 1

        runs.append({
            "label": run_dir.name,
            "index": run_index,
            "weight": weight_path
        })

    return sorted(runs, key=lambda r: r["index"])


@st.cache_resource(show_spinner=False)
def load_model(weight_path: str, conf: float, device: str | None = None):
    return AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=weight_path,
        confidence_threshold=conf,
        device=device,
    )


def _image_from_bytes(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def _to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _draw_result(image: Image.Image, result) -> Image.Image:
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    colors = {
        "Dirt-Wire": "#0077b6",  # ocean blue
        "Critical": "red",
    }
    for obj in result.object_prediction_list:
        x, y, w, h = obj.bbox.to_xywh()
        x2, y2 = x + w, y + h
        color = colors.get(obj.category.name, "red")
        draw.rectangle([(x, y), (x2, y2)], outline=color, width=2)
        label = f"{obj.category.name} {obj.score.value:.2f}"
        if font:
            draw.text((x, y), label, fill=color, font=font)
        else:
            draw.text((x, y), label, fill=color)
    return image


def _format_badge(label: str, count: int, color: str) -> str:
    badge_color = color if count > 0 else "#6c757d"
    return f"<span style='color:{badge_color}; font-weight:bold'>{label}: {count}</span>"


def _demo_images():
    samples = []
    for path in DEMO_IMAGE_PATHS:
        try:
            data = Path(path).resolve().read_bytes()
            samples.append((Path(path).name, data))
        except FileNotFoundError:
            continue
    return samples


def _annotate_image(model, data: bytes, slice_h: int, slice_w: int, overlap: float):
    image = _image_from_bytes(data)
    result = get_sliced_prediction(
        image,
        model,
        slice_height=slice_h,
        slice_width=slice_w,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
        verbose=0,
    )
    annotated = _draw_result(image.copy(), result)
    return annotated, result


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
st.caption("Upload a single image or a .zip folder of images to preview SAHI-sliced YOLO predictions with bounding boxes.")

runs = list_available_runs()
if not runs:
    st.error("No trained runs found under models. Train a model first to enable the demo.")
    st.stop()

run_labels = [run["label"] for run in runs]
selected_run = st.sidebar.selectbox("Model checkpoint", run_labels, index=len(run_labels) - 1)
selected_weight = next(run for run in runs if run["label"] == selected_run)["weight"]

confidence = st.sidebar.slider("Confidence threshold", min_value=0.05, max_value=0.95, value=0.5, step=0.05)
slice_height = st.sidebar.number_input("Slice height", min_value=64, max_value=1024, value=256, step=32)
slice_width = st.sidebar.number_input("Slice width", min_value=64, max_value=1024, value=256, step=32)
slice_overlap = st.sidebar.slider("Slice overlap", min_value=0.0, max_value=0.9, value=0.2, step=0.05)
max_images = st.sidebar.slider("Max images to process", min_value=1, max_value=25, value=10)

model = load_model(str(selected_weight), conf=confidence)

input_mode = st.radio(
    "What would you like to upload?",
    ("Single images", "Folder (.zip)"),
    help="Select single images (you can pick multiple files) or upload a zipped folder."
)

if "pending_images" not in st.session_state:
    st.session_state["pending_images"] = []

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
            st.session_state["pending_images"].append((file.name, data))
else:
    uploaded_zip = st.file_uploader("Upload a .zip archive", type=["zip"])
    if uploaded_zip is not None:
        uploaded_zip.seek(0)
        extracted = _extract_images_from_zip(uploaded_zip.read())
        if extracted:
            st.session_state["pending_images"].extend(extracted)
        else:
            st.warning("No supported image files were found inside the archive.")

demo_clicked = st.button("🟡 Load demo images", help="Add the configured sample images to the queue for a quick test.")
if demo_clicked:
    demo_imgs = _demo_images()
    if demo_imgs:
        st.session_state["pending_images"].extend(demo_imgs)
        st.success(f"Loaded {len(demo_imgs)} demo image(s).")
    else:
        st.warning("No demo images found. Update DEMO_IMAGE_PATHS with valid jpg/png files.")

if st.session_state["pending_images"]:
    pending_images = st.session_state["pending_images"][:max_images]
    st.info(f"Ready to process {len(pending_images)} image(s). (Showing up to {max_images})")
else:
    pending_images = []
    st.warning("Add some images to start.")

run_inference = st.button("Generate predictions", disabled=not pending_images)

if run_inference and pending_images:
    with st.spinner("Running inference..."):
        for name, data in pending_images:
            try:
                start = time.perf_counter()
                annotated, result = _annotate_image(model, data, slice_height, slice_width, slice_overlap)
                elapsed_ms = (time.perf_counter() - start) * 1000
            except Exception as exc:  # noqa: BLE001 - present errors to the user
                st.error(f"Failed to process {name}: {exc}")
                continue

            orig_img = _image_from_bytes(data)
            cols = st.columns([1, 1])
            with cols[0]:
                st.image(orig_img, caption=f"Original · {name}", use_container_width=True)
            with cols[1]:
                detections = len(result.object_prediction_list)
                st.image(annotated, caption=f"Predictions ({detections} boxes) · {name}", use_container_width=True)

                crit_count = sum(1 for obj in result.object_prediction_list if obj.category.name == "Critical")
                dirt_count = sum(1 for obj in result.object_prediction_list if obj.category.name == "Dirt-Wire")
                st.markdown(
                    _format_badge("Critical", crit_count, "red")
                    + "<br/>"
                    + _format_badge("Dirt-Wire", dirt_count, "#0077b6"),
                    unsafe_allow_html=True,
                )
                st.caption(f"Inference time: {elapsed_ms:.1f} ms")

            st.markdown(
                f"[Open original full size]({_to_data_url(orig_img)}) · "
                f"[Open annotated full size]({_to_data_url(annotated)})",
                help="Opens the images in a new tab so you can zoom in deeply.",
            )

    st.success("Prediction completed.")
    
