#!/bin/bash

# === Config ===
# We now use the local API endpoint which handles saving
API_URL="http://localhost:5001/capture"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

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
# Allow 'low_confidence' or other labels if needed, but primarily open/closed
if [[ -z "$LABEL" ]]; then
    echo "Label cannot be empty."
    exit 1
fi

echo "Capturing image for label: $LABEL"

# === Call API ===
response=$(curl -s "$API_URL?status=$LABEL")

# === Check result ===
# Simple check if "success" is in the response
if [[ "$response" == *"success"* ]]; then
    echo "✅ Image captured successfully."
    echo "Server response: $response"
else
    echo "❌ Failed to capture image."
    echo "Server response: $response"
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

