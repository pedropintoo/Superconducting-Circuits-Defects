
## Setup Label Studio

```bash
python3 -m venv labelstudio-env
source labelstudio-env/bin/activate
pip install -r requirements.txt
```

## Run Label Studio

> Important: To disable analytics tracking, set all these environment variables to empty strings or `False`!

Note: tested with Ubuntu 24.04 and Label Studio 1.12.0!

```bash
export FRONTEND_SENTRY_DSN=""
export SENTRY_DSN=""
export COLLECT_ANALYTICS=False

export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=$HOME

label-studio --host 0.0.0.0 --port 8080 start
```

## Importing data

To easily work with a big number of files, it is recommended to use the "Local Storage" option when creating a new project.
[Documentation](https://labelstud.io/guide/storage#Local-storage)

Intuition: you just need to create a project, after that, you go to the "Settings" tab, then to "Storage" and create a new "Local Storage" pointing to the folder where your data is stored.

## Working together in the same project

Then, other users can access the project by going to `http://<ip_address>:8080` in their web browser.  

## Saving and exporting annotations

Important: to recreate the Label Studio don't forget to export in JSON format!! (see `/backup` folder)

To train a YOLO model, you need to export as "YOLO format", and pass it through the `process-labeling.py` script.

## Help tutorial

https://www.youtube.com/watch?v=R1ozTMrujOE


## Using ML backend with Label Studio

Inside `ml_labeling_backend/` you can find instructions on how to set up and use the ML backend with Label Studio.
You just need to run with:

```bash
docker-compose up --build
```
This will start the ML backend server that will connect to your Label Studio instance.
