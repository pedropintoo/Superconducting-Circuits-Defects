
## Setup Label Studio

```bash
python3 -m venv labelstudio-env
source labelstudio-env/bin/activate
pip install -r requirements.txt
```

## Run Label Studio

> Important: To disable analytics tracking, set all these environment variables to empty strings or `False`!
```bash
FRONTEND_SENTRY_DSN="" SENTRY_DSN="" COLLECT_ANALYTICS=False label-studio
```
