import numpy as np
import tensorflow as tf
import os
import json
from PIL import Image

# === Load ROI from JSON ===
def load_roi(filepath="roi.json"):
    with open(filepath, "r") as f:
        roi_data = json.load(f)
    return roi_data["x"], roi_data["y"], roi_data["x1"], roi_data["y1"]

ROI = load_roi()

def crop_and_preprocess(img_path, target_size=(96, 96)):
    img = Image.open(img_path).convert('L') # Convert to Grayscale
    x, y, x1, y1 = ROI
    img = img.crop((x, y, x1, y1))
    img = img.resize(target_size)
    return np.array(img).astype('float32') / 255.0

def verify_model(model_path, data_dir):
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    
    classes = ['closed', 'open']
    results = []

    print(f"Verifying model: {model_path}")
    for label, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, 'train', class_name)
        if not os.path.exists(class_dir):
            continue
            
        count = 0
        correct = 0
        for img_name in os.listdir(class_dir):
            if img_name.endswith('.jpg'):
                img_path = os.path.join(class_dir, img_name)
                img_array = crop_and_preprocess(img_path)
                
                # Check if input is quantized
                if input_details[0]['dtype'] == np.int8:
                    input_scale, input_zero_point = input_details[0]['quantization']
                    img_array = (img_array / input_scale + input_zero_point).astype(np.int8)
                
                input_data = img_array.reshape(input_shape)
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()

                output_data = interpreter.get_tensor(output_details[0]['index'])
                predicted_label = np.argmax(output_data)
                
                if predicted_label == label:
                    correct += 1
                count += 1
                
        if count > 0:
            accuracy = correct / count
            print(f"Class {class_name.upper()}: {correct}/{count} correct ({accuracy*100:.2f}%)")
            results.append(accuracy)

    if results:
        print(f"Overall Accuracy: {np.mean(results)*100:.2f}%")

if __name__ == "__main__":
    verify_model("gate_detector_tiny.tflite", "./data")
