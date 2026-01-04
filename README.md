# ESP32 Gate Detector with TinyML

This project implements a smart gate detector using an ESP32S3 and a PC-based server for training and verification. It uses a TinyCNN model to classify the gate as **OPEN** or **CLOSED**. The system is also integrated with HomeKit (using HomeSpan library) for easy monitoring and status checks.

## Project Structure

- **`esp32gate/`**: Arduino sketch for the ESP32 (or XIAO ESP32S3). Runs inference using TFLite Micro.
- **`server/`**: Python scripts for data collection, model training, conversion, and a Flask server sibling.
- **`tester/`**: (Optional) Sandbox for testing sensor/camera components.

## Setup Guide

### 0. Clone the Repository
Start by cloning the repository to your local machine:

```bash
git clone https://github.com/rightedges/esp32gate.git
cd esp32gate
```

To update the repository to the latest version, run:
```bash
git pull
```

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
pip install -r requirements.txt
```
*(Alternatively, you can manually install: `pip install tensorflow tflite-runtime flask pillow numpy`)*

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

### Auto-Start Server on Reboot
We provide a startup script `server/start_gate_server.sh` that can be used to automatically start the server on boot.

To set this up, add a LINE to your user's `crontab`:

1. Run `crontab -e`
2. Add the following line:
```
@reboot /path/to/esp32gate/server/start_gate_server.sh >> /path/to/esp32gate/server/cron_log.txt 2>&1
```
*(Replace `/path/to/esp32gate` with the actual absolute path to your project directory).*

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
# esp32gate
