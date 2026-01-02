import io
import json
import numpy as np
try:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        Interpreter = tflite.Interpreter
    except ImportError:
        # Fallback will be handled in main code or let it crash if essential
        print("Warning: Neither tensorflow nor tflite_runtime installed.")
        Interpreter = None
from PIL import Image
from flask import Flask, request, Response
import urllib.request
import base64
import logging
from datetime import datetime

# === Setup Logging ===
logging.basicConfig(
    filename='gate_status.log',
    level=logging.INFO,
    format='%(asctime)s gate-detector: %(message)s',
    datefmt='%b %d %H:%M:%S'
)

app = Flask(__name__)

# === Load ROI from JSON ===
def load_roi(filepath="roi.json"):
    """Loads the region of interest (ROI) from a JSON file."""
    try:
        with open(filepath, "r") as f:
            roi_data = json.load(f)
        return roi_data["x"], roi_data["y"], roi_data["x1"], roi_data["y1"]
    except FileNotFoundError:
        print(f"Error: ROI file not found at {filepath}. Using default ROI (0,0,640,480).")
        return 0, 0, 640, 480

ROI = load_roi()

# === Crop and Preprocess function ===
def crop_and_preprocess(image: Image.Image, target_size=(48, 48)):
    """Crops the input image to the ROI and preprocesses it for the model."""
    # Convert to grayscale
    img = image.convert('L')
    
    # Crop
    x, y, x1, y1 = ROI
    img = img.crop((x, y, x1, y1))
    
    # Resize (Use BOX or BILINEAR to match training/ESP32)
    resample_filter = Image.BOX if hasattr(Image, 'BOX') else Image.BILINEAR
    img = img.resize(target_size, resample=resample_filter)
    
    # Normalize to 0-1 float32
    img_array = np.array(img).astype('float32') / 255.0
    
    # Add batch and channel dimensions: (1, 48, 48, 1)
    img_array = img_array.reshape(1, 48, 48, 1)
    return img_array

# === Model ===
MODEL_PATH = "gate_detector_tiny.tflite"

def load_interpreter(model_path):
    """Loads the TFLite interpreter."""
    if Interpreter is None:
        return None
    try:
        interpreter = Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        print(f"Loaded TFLite model from {model_path}")
        return interpreter
    except Exception as e:
        print(f"Error loading TFLite model: {e}")
        return None

interpreter = load_interpreter(MODEL_PATH)
if interpreter:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
else:
    print("CRITICAL: Failed to load TFLite interpreter.")

# === Camera URL and Credentials ===
gate_url = 'http://192.168.50.82/ISAPI/ContentMgmt/StreamingProxy/channels/801/picture?cmd=refresh'
gate_user = 'admin'
gate_password = 'pccw1234'
auth_str = f'{gate_user}:{gate_password}'.encode('ascii')
auth_b64 = base64.b64encode(auth_str).decode('ascii')
headers = {'Authorization': 'Basic ' + auth_b64}

def get_camera_image():
    """Fetches an image from the camera."""
    try:
        req = urllib.request.Request(gate_url, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as response:
            img_data = response.read()
        return Image.open(io.BytesIO(img_data))
    except Exception as e:
        print(f"Error fetching image from camera: {e}")
        return None

def predict_gate_status(image):
    """Predicts the gate status (open/closed) using TFLite model."""
    if image is None:
        return "error"
    
    if interpreter is None:
        return "error - model not loaded"

    try:
        # Preprocess
        input_data = crop_and_preprocess(image)
        
        # Check input details (for debugging or if model changes)
        # input_details[0]['index'] is the input tensor index
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        # Output is likely [1, 2] probability scores (softmax)
        
        predicted_class = np.argmax(output_data)
        
        # Classes: 0=closed, 1=open
        return "closed" if predicted_class == 0 else "open"
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return "error"

@app.route("/", methods=['GET'])
def get_gate_status():
    """
    Web service endpoint that returns the gate status as JSON.
    """
    image = get_camera_image()
    status = predict_gate_status(image)
    
    if "error" in status:
        return Response(json.dumps({"status": "error", "message": "Failed to retrieve image or predict"}),
                        status=500, mimetype='application/json')

    log_msg = f"Gate status: {status}"
    logging.info(log_msg)

    response_data = {"status": status}
    return Response(json.dumps(response_data), mimetype='application/json')

if __name__ == "__main__":
    # Use 0.0.0.0 to listen on all interfaces
    app.run(host='0.0.0.0', port=5001, debug=False)
