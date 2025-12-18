
## Purpose
Label, standardize, and serve annotations for the superconducting defect dataset. This guide covers spinning up Label Studio, connecting storage, backing up projects, and attaching the YOLO+SAHI model backend.

## Setup Label Studio
- Create an isolated environment and install requirements:
```bash
python3 -m venv labelstudio-env
source labelstudio-env/bin/activate
pip install -r requirements.txt
```

## Run Label Studio
> Disable analytics: set these to empty or False. (Very important since we were under NDA when developing.)

Tested with Ubuntu 24.04 and Label Studio 1.12.0.

```bash
export FRONTEND_SENTRY_DSN=""
export SENTRY_DSN=""
export COLLECT_ANALYTICS=False

export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=$HOME

label-studio --host 0.0.0.0 --port 8080 start
```
Or use the helper script: [start-label-studio.sh](start-label-studio.sh).

## Database and backups
- Default storage is SQLite under the Label Studio user data directory; for quick setups this is fine.
- Back up a project by exporting annotations (JSON) from the UI and by copying the Label Studio data directory if you want the database state. A curated export is kept under [backup/](backup).

## Connect the ML model
- Start the backend (Docker or scripted) as described in [model_backend/README.md](model_backend/README.md).
- In Label Studio: Settings → Machine Learning → Add Model, set the backend URL (default http://localhost:9090), then Save and Sync.

## Connect storage (import data)

To easily work with a big number of files, it is recommended to use the "Local Storage" option when creating a new project.
[Documentation](https://labelstud.io/guide/storage#Local-storage)

Intuition: you just need to create a project, after that, you go to the "Settings" tab, then to "Storage" and create a new "Local Storage" pointing to the folder where your data is stored.

## Working together in the same project

Then, other users can access the project by going to http://<ip_address>:8080 in their web browser.  

## Saving and exporting annotations

Important: to recreate the Label Studio don't forget to export in JSON format!! (see [backup/](backup))

To train a YOLO model, you need to export as "YOLO format", and pass it through the [process-labeling.py](process-labeling.py) script.

## Help tutorial

https://www.youtube.com/watch?v=R1ozTMrujOE

### Notes
- Default DB is SQLite under the Label Studio data dir; for backups, export JSON from the UI and copy the data dir. Curated exports live in [backup/](backup).
- Ensure CUDA toolkit is installed if you want GPU acceleration when running the backend script.
- Backend builds on the Label Studio ML template with a customized model.py for YOLO+SAHI; see [model_backend/README.md](model_backend/README.md).
