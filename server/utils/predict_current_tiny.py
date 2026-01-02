import io
import json
import numpy as np
import tensorflow as tf
from PIL import Image
import urllib.request
import base64
import os

# === Configuration ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROI_PATH = os.path.join(BASE_DIR, "roi.json")
MODEL_PATH = os.path.join(BASE_DIR, "gate_detector_tiny.tflite")

# Camera details
gate_url = 'http://192.168.50.82/ISAPI/ContentMgmt/StreamingProxy/channels/801/picture?cmd=refresh'
gate_user = 'admin'
gate_password = 'pccw1234'

# === Load ROI from JSON ===
def load_roi(filepath=ROI_PATH):
    with open(filepath, "r") as f:
        roi_data = json.load(f)
    return roi_data["x"], roi_data["y"], roi_data["x1"], roi_data["y1"]

ROI = load_roi()

def fetch_image():
    """Fetches the current image from the camera."""
    auth_str = f'{gate_user}:{gate_password}'.encode('ascii')
    auth_b64 = base64.b64encode(auth_str).decode('ascii')
    headers = {'Authorization': 'Basic ' + auth_b64}
    
    req = urllib.request.Request(gate_url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            return Image.open(io.BytesIO(response.read()))
    except Exception as e:
        print(f"Error fetching image: {e}")
        return None

def preprocess_image(image, target_size=(96, 96)):
    """Crops and resizes the image for the TinyML model."""
    img = image.convert('L') # Grayscale
    x, y, x1, y1 = ROI
    img = img.crop((x, y, x1, y1))
    
    # Use BOX to match training and ESP32 logic
    resample_filter = Image.BOX if hasattr(Image, 'BOX') else Image.BILINEAR
    img = img.resize(target_size, resample=resample_filter)
    
    arr = np.array(img).astype('float32')
    print(f"Python ROI Avg Brightness: {np.mean(arr):.2f}")
    return arr / 255.0

def predict_current():
    image = fetch_image()
    if image is None:
        return

    print(f"Captured image resolution: {image.size}")
    img_array = preprocess_image(image)

    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Check if input is quantized
    if input_details[0]['dtype'] == np.int8:
        input_scale, input_zero_point = input_details[0]['quantization']
        img_array = (img_array / input_scale + input_zero_point).astype(np.int8)

    input_data = img_array.reshape(input_details[0]['shape'])
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Dequantize output if necessary
    if output_details[0]['dtype'] == np.int8:
        output_scale, output_zero_point = output_details[0]['quantization']
        output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale

    # Since there's a softmax layer, values are probabilities
    predicted_label = np.argmax(output_data)
    confidence = output_data[0][predicted_label]
    
    classes = ['CLOSED', 'OPEN']
    status = classes[predicted_label]
    
    print(f"Current Gate Status: {status}")
    print(f"Confidence Level: {confidence * 100:.2f}%")

if __name__ == "__main__":
    predict_current()
