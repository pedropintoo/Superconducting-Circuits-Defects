#!/bin/bash
#
# Script to start Label Studio with specific configurations for local file serving.
# 
# Requirements:
# - Install requirements.txt in a python virtual environment.
#
# Usage: ./start-label-studio.sh
#
# Authors: Pedro Pinto, João Pinto, Fedor Chikhachev

export FRONTEND_SENTRY_DSN=""
export SENTRY_DSN=""
export COLLECT_ANALYTICS=False

export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=$HOME

label-studio --host 0.0.0.0 --port 8080 start
