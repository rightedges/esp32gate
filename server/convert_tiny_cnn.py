import tensorflow as tf
import numpy as np
import os
import json
from PIL import Image

# === Load ROI from JSON ===
def load_roi(filepath="roi.json"):
    with open(filepath, "r") as f:
        roi_data = json.load(f)
    return roi_data["x"], roi_data["y"], roi_data["x1"], roi_data["y1"]

ROI = load_roi()

def crop_and_preprocess(img_path, target_size=(48, 48)):
    img = Image.open(img_path).convert('L')
    x, y, x1, y1 = ROI
    img = img.crop((x, y, x1, y1))
    img = img.resize(target_size)
    return np.array(img).astype('float32') / 255.0

def representative_dataset():
    dataset_path = "./data/train"
    images = []
# Function to load dataset (assuming it loads and preprocesses images)
def load_dataset(dataset_dir="./data", target_size=(48, 48)):
    x_data = []
    y_data = [] # Assuming labels are not needed for representative dataset
    
    # Collect images from 'train' subdirectory
    train_dir = os.path.join(dataset_dir, "train")
    if not os.path.exists(train_dir):
        print(f"Warning: Training directory '{train_dir}' not found. Representative dataset might be empty.")
        return np.array([]), np.array([])

    for root, _, files in os.walk(train_dir):
        for file in files:
            if file.endswith(".jpg"):
                img_path = os.path.join(root, file)
                img = crop_and_preprocess(img_path, target_size)
                x_data.append(img.reshape(target_size[0], target_size[1], 1)) # Add channel dimension
                if len(x_data) >= 100: # Limit to 100 samples for representative dataset
                    break
        if len(x_data) >= 100:
            break
            
    return np.array(x_data), np.array(y_data) # Return as numpy arrays

# === Model Conversion ===

# Load model from .keras file
model = tf.keras.models.load_model("gate_detector_tiny.keras")

# Force static batch size using concrete function
@tf.function(input_signature=[tf.TensorSpec(shape=[1, 48, 48, 1], dtype=tf.float32)])
def model_static(x):
    return model(x)

concrete_func = model_static.get_concrete_function()

# Use training data for representative dataset
x_samples, _ = load_dataset("./data")
def representative_dataset():
    for i in range(min(100, len(x_samples))):
        yield [x_samples[i:i+1].astype(np.float32)]

# Create TFLite Converter from concrete function
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.representative_dataset = representative_dataset
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.inference_input_type = tf.int8
# converter.inference_output_type = tf.int8
# Removed explicit Int8 I/O for better compatibility with TFLM wrappers

tflite_model = converter.convert()

# Save TFLite model
with open("gate_detector_tiny.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved as gate_detector_tiny.tflite")

# Convert TFLite to C Header
def hex_to_c_array(hex_data, var_name):
    c_str = f"const unsigned char {var_name}[] = {{\n  "
    line_bytes = 12
    for i, b in enumerate(hex_data):
        c_str += f"0x{b:02x}, "
        if (i + 1) % line_bytes == 0:
            c_str += "\n  "
    c_str = c_str.rstrip(", \n") + "\n};\n"
    c_str += f"const int {var_name}_len = {len(hex_data)};\n"
    return c_str

c_array_content = hex_to_c_array(tflite_model, "gate_detector_model")

with open("model_data.h", "w") as f:
    f.write("#ifndef MODEL_DATA_H\n")
    f.write("#define MODEL_DATA_H\n\n")
    f.write(c_array_content)
    f.write("\n#endif // MODEL_DATA_H\n")

print("C header saved as model_data.h")
