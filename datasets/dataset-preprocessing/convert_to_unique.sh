#!/bin/bash

# Base directory
BASE_DIR="Second_Batch"
OUTPUT_DIR="full_dataset"

# Check if the provided argument is a directory
if [ ! -d "$BASE_DIR" ]; then
    echo "Error: $BASE_DIR is not a directory."
    exit 1
fi

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Function to process directories recursively
process_directory() {
    local DIR="$1"
    local RELATIVE_PATH="${DIR#$BASE_DIR/}"

    # Find all .jpg files in the current directory
    find "$DIR" -maxdepth 1 -type f -name "*.jpg" | while read -r FILE; do
        ORIGINAL_NAME="$(basename "$FILE" .jpg)"
        SUBDIR_PATH="${RELATIVE_PATH//\//-}"
        BASE_PATH="${BASE_DIR//\//-}"
        NEW_NAME="${BASE_PATH}-${SUBDIR_PATH}-${ORIGINAL_NAME}.jpg"
        cp "$FILE" "$OUTPUT_DIR/$NEW_NAME"
    done

    # Process subdirectories
    find "$DIR" -mindepth 1 -maxdepth 1 -type d | while read -r SUBDIR; do
        process_directory "$SUBDIR"
    done
}

# Start processing from the base directory
process_directory "$BASE_DIR"