import argparse
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
        print("Error: neither 'tensorflow' nor 'tflite_runtime' is installed.")
        print("Please install one of them: pip install tensorflow OR pip install tflite-runtime")
        import sys
        sys.exit(1)
from PIL import Image
import urllib.request
import base64
import os
import sys

# === Configuration ===
MODEL_PATH = "gate_detector_tiny.tflite"
ROI_PATH = "roi.json"
GATE_URL = 'http://192.168.50.82/ISAPI/ContentMgmt/StreamingProxy/channels/801/picture?cmd=refresh'
GATE_USER = 'admin'
GATE_PASSWORD = 'pccw1234'

def load_roi(filepath=ROI_PATH):
    try:
        with open(filepath, "r") as f:
            roi_data = json.load(f)
        return roi_data["x"], roi_data["y"], roi_data["x1"], roi_data["y1"]
    except Exception as e:
        print(f"Error loading ROI: {e}")
        return 0, 0, 640, 480

ROI = load_roi()

def get_camera_image():
    print(f"Fetching image from {GATE_URL}...")
    auth_str = f'{GATE_USER}:{GATE_PASSWORD}'.encode('ascii')
    auth_b64 = base64.b64encode(auth_str).decode('ascii')
    headers = {'Authorization': 'Basic ' + auth_b64}
    
    try:
        req = urllib.request.Request(GATE_URL, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as response:
            return Image.open(io.BytesIO(response.read()))
    except Exception as e:
        print(f"Error fetching camera image: {e}")
        return None

def preprocess_image(image, target_size=(48, 48)):
    # Convert to grayscale
    img = image.convert('L')
    
    # Crop
    x, y, x1, y1 = ROI
    print(f"Cropping to ROI: {ROI}")
    img = img.crop((x, y, x1, y1))
    
    # Resize
    resample_filter = Image.BOX if hasattr(Image, 'BOX') else Image.BILINEAR
    img = img.resize(target_size, resample=resample_filter)
    
    # Normalize
    img_array = np.array(img).astype('float32') / 255.0
    return img_array.reshape(1, 48, 48, 1)

def run_inference(image):
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found.")
        return

    # Load TFLite model
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocess
    input_data = preprocess_image(image)
    
    # Set input
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run
    interpreter.invoke()
    
    # Get output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(f"Raw Output: {output_data}")
    
    predicted_index = np.argmax(output_data)
    confidence = output_data[0][predicted_index]
    
    status = "CLOSED" if predicted_index == 0 else "OPEN"
    print(f"Prediction: \033[1m{status}\033[0m (Confidence: {confidence:.4f})")

def main():
    parser = argparse.ArgumentParser(description="Test Gate Detector TFLite Model")
    parser.add_argument("--image", type=str, help="Path to local image file (optional)")
    args = parser.parse_args()

    image = None
    if args.image:
        if os.path.exists(args.image):
            print(f"Loading local image: {args.image}")
            image = Image.open(args.image)
        else:
            print(f"Error: File not found: {args.image}")
            sys.exit(1)
    else:
        image = get_camera_image()

    if image:
        run_inference(image)
    else:
        print("Failed to obtain image.")

if __name__ == "__main__":
    main()
