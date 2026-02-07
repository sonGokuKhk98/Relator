#!/usr/bin/env bash
# Render build script — installs deps + downloads production.db from Google Drive
set -e

echo "=== Installing Python dependencies ==="
pip install -r requirements.txt

# Download production.db from Google Drive if not already present
if [ ! -f "production.db" ]; then
    echo "=== Downloading production.db from Google Drive ==="

    if [ -z "$GDRIVE_FILE_ID" ]; then
        echo "ERROR: GDRIVE_FILE_ID env var not set. Cannot download production.db."
        echo "Set it in Render dashboard -> Environment -> GDRIVE_FILE_ID"
        exit 1
    fi

    # Install gdown for Google Drive downloads
    pip install gdown

    # Download using gdown (handles large files + confirmation prompts)
    gdown "https://drive.google.com/uc?id=${GDRIVE_FILE_ID}" -O production.db

    echo "=== production.db downloaded ($(du -h production.db | cut -f1)) ==="
else
    echo "=== production.db already exists ($(du -h production.db | cut -f1)), skipping download ==="
fi
