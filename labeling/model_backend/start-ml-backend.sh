#!/bin/bash
#
# Script to start Label Studio ML Backend with specific configurations.
# 
# Requirements:
# - Install requirements.txt in a python virtual environment.
#
# Usage: ./start-ml-backend.sh
#
# Authors: Pedro Pinto, João Pinto, Fedor Chikhachev

CONFIDENCE_FACTOR=0.5 SLICE_HEIGHT=256 SLICE_WIDTH=256 SLICE_OVERLAP=0.2 \
    label-studio-ml start .
