# Label Studio ML Backend for YOLO+SAHI Pre-annotations

YOLO+SAHI ML backend for Label Studio pre-annotations (Critical, Dirt-Wire). Start via Docker or shell, then register it under Settings → Machine Learning in Label Studio.

## Run without Docker (shell)
Use a dedicated venv. Enable CUDA if you want GPU inference.
```bash
python -m venv ml-backend
source ml-backend/bin/activate
pip install -r requirements.txt

./start-ml-backend.sh
```

Or, running without bash script:

```bash
CONFIDENCE_FACTOR=0.5 SLICE_HEIGHT=256 SLICE_WIDTH=256 SLICE_OVERLAP=0.2 \
label-studio-ml start .
```

Connect in Label Studio: Settings → Machine Learning → Add Model → URL http://localhost:9090 → Save → Sync.

Check it's running:
```bash
curl http://localhost:9090
```

# Configuration

The `model.py` file contains the main logic for the ML backend.
Inside `model.py`, you can adjust the following parameters to fit your needs:
- `CONFIDENCE_FACTOR`: Confidence threshold for predictions. (default: 0.5)
- `SLICE_HEIGHT`: Height of the image slices. (default: 256)
- `SLICE_WIDTH`: Width of the image slices. (default: 256)
- `SLICE_OVERLAP`: Overlap percentage between slices. (default: 0.2)

## Customization
Built on the Label Studio ML template with a customized `model.py` for YOLO+SAHI. You can further adjust logic or swap the model under `./dir_with_your_model`.
