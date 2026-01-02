#!/bin/bash
# Description: Automates the retraining, conversion, and deployment of the TinyCNN model.

echo "=== Starting Retraining Pipeline ==="

# Detect Python Environment
if [ -f "./tiny-env/bin/python3" ]; then
    PYTHON_CMD="./tiny-env/bin/python3"
    echo "Using tiny-env: $PYTHON_CMD"
elif [ -f "./venv_new/bin/python3" ]; then
    PYTHON_CMD="./venv_new/bin/python3"
    echo "Using venv_new: $PYTHON_CMD"
else
    PYTHON_CMD="python3"
    echo "Using system python: $PYTHON_CMD"
fi

# 1. Train Model
echo ">>> Training Model..."
$PYTHON_CMD train_tiny_cnn.py
if [ $? -ne 0 ]; then
    echo "❌ Training failed. Aborting."
    exit 1
fi

# 2. Convert Model (to TFLite and C Header)
echo ">>> Converting Model..."
$PYTHON_CMD convert_tiny_cnn.py
if [ $? -ne 0 ]; then
    echo "❌ Conversion failed. Aborting."
    exit 1
fi

# 3. Deploy Header to ESP32 Project
echo ">>> Deploying to ESP32..."
if [ -d "../esp32gate" ]; then
    cp model_data.h ../esp32gate/model_data.h
    echo "✅ Copied model_data.h to ../esp32gate/"
else
    echo "⚠️ Warning: ../esp32gate directory not found. Skipping deployment to ESP32."
fi

echo "=== Pipeline Complete! ==="
echo "New model is ready at: server/gate_detector_tiny.tflite"
echo "New C header is ready at: esp32gate/model_data.h"
