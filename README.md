# ESP32 Gate Detector with TinyML

This project implements a smart gate detector using an ESP32S3 and a PC-based server for training and verification. It uses a TinyCNN model to classify the gate as **OPEN** or **CLOSED**.

## Project Structure

- **`esp32gate/`**: Arduino sketch for the ESP32 (or XIAO ESP32S3). Runs inference using TFLite Micro.
- **`server/`**: Python scripts for data collection, model training, conversion, and a Flask server sibling.
- **`tester/`**: (Optional) Sandbox for testing sensor/camera components.

## Setup Guide

### 1. Hardware
- **Board**: **Seeed Studio XIAO ESP32S3 (Sense)** (Highly Recommended).
    - **Requirement**: Must have **PSRAM** (e.g., 8MB) enabled to run the TinyCNN model effectively.
    - *Note: Standard ESP32/ESP32-CAM boards may struggle with memory.*
- **Camera**: OV2640 (standard) or compatible camera module.
- **Wiring**: Ensure correct pin definitions in `esp32gate/DEV_DoorsWindows.h` or `esp32gate.ino`.

### 2. Server Environment (Python)

Navigate to the `server/` directory and set up a virtual environment.

```bash
cd server
python3 -m venv tiny-env
source tiny-env/bin/activate
pip install tensorflow tflite-runtime flask pillow numpy
```
*(Note: You can use `requirements.txt` if available, or just install the core deps above).*

### 3. ESP32 Firmware

1. Open `esp32gate/esp32gate.ino` in Arduino IDE.
2. Install necessary libraries (e.g., `TensorFlowLite_ESP32`, `ESP32 Camera`).
3. Select your board (e.g., "XIAO_ESP32S3") and configure PSRAM (OPI PSRAM).
4. Upload the sketch.

## Usage

### Testing the Server Model
You can use the server scripts to check the model against the live camera feed without the ESP32.

```bash
cd server
./tiny-env/bin/python3 test_model.py
```

### Running the Flask Server
The `gate.py` script runs a web server that exposes the gate status.

```bash
cd server
./tiny-env/bin/python3 gate.py
```

## Retraining the Model

We have automated the process of collecting data, retraining, and updating the ESP32 firmware code.

### Step 1: Collect Data
Run the labeling script to capture images and label them.

```bash
cd server
./label_and_capture.sh
```
*Follow the prompts. You can choose to "Open" or "Closed".*

### Step 2: Retrain & Deploy
The script above will ask if you want to retrain immediately. If you say **Yes**, it will:
1. Train a new TinyCNN model.
2. Convert it to TFLite.
3. Convert it to a C header file (`model_data.h`).
4. Copy `model_data.h` to `../esp32gate/`.

You can also run this manually:
```bash
cd server
./retrain_deploy.sh
```

### Step 3: Flash ESP32
After the `model_data.h` is updated, re-upload the `esp32gate` sketch to your board to apply the new model.

## Troubleshooting
- **Import Error (tensorflow)**: Ensure you are using the virtual environment (`source tiny-env/bin/activate`).
- **Camera Error**: Check the URL and credentials in `gate.py` and `label_and_capture.sh`.
