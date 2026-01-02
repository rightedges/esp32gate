#!/bin/bash

# === Config ===
BASE_DIR="./data/train"
GATE_URL="http://192.168.50.82/ISAPI/ContentMgmt/StreamingProxy/channels/801/picture?cmd=refresh"
GATE_USER="admin"
GATE_PASSWORD="pccw1234"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FILENAME="${TIMESTAMP}.jpg"

# === Parse argument ===
LABEL=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --label)
            LABEL="$2"
            shift 2
            ;;
        *)
            echo "❌ Unknown argument: $1"
            echo "Usage: $0 [--label open|closed]"
            exit 1
            ;;
    esac
done

# === Ask interactively if no label provided ===
if [[ -z "$LABEL" ]]; then
    read -rp "Is the gate OPEN or CLOSED? (Type 'open' or 'closed'): " LABEL
fi

# === Validate label ===
if [[ "$LABEL" != "open" && "$LABEL" != "closed" ]]; then
    echo "Invalid label. Please use 'open' or 'closed'."
    exit 1
fi

SAVE_DIR="$BASE_DIR/$LABEL"

# === Ensure directory exists ===
mkdir -p "$SAVE_DIR"

# === Capture and save ===
curl -u "$GATE_USER:$GATE_PASSWORD" -s "$GATE_URL" -o "$SAVE_DIR/$FILENAME"

# === Result ===
if [ -f "$SAVE_DIR/$FILENAME" ]; then
    echo "Image saved to: $SAVE_DIR/$FILENAME"
else
    echo "Failed to capture or save the image."
    exit 1
fi

# === Optional Retraining ===
echo ""
echo "Do you want to retrain the model and deploy to ESP32 now? (y/N)"
read -r RETRAIN_CHOICE
if [[ "$RETRAIN_CHOICE" =~ ^[Yy]$ ]]; then
    ./retrain_deploy.sh
else
    echo "Skipping retraining."
fi

