
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

label-studio
```

## Importing data

To easily work with a big number of files, it is recommended to use the "Local Storage" option when creating a new project.
[Documentation](https://labelstud.io/guide/storage#Local-storage)

Intuition: you just need to create a project, after that, you go to the "Settings" tab, then to "Storage" and create a new "Local Storage" pointing to the folder where your data is stored.

## Help tutorial

https://www.youtube.com/watch?v=R1ozTMrujOE
