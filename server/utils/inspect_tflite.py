import tensorflow as tf
import numpy as np

model_path = "gate_detector_tiny.tflite"

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("=== Input Details ===")
print(f"Shape: {input_details[0]['shape']}")
print(f"Dtype: {input_details[0]['dtype']}")
print(f"Index: {input_details[0]['index']}")
print(f"Quantization: {input_details[0]['quantization']}")

print("\n=== Output Details ===")
print(f"Shape: {output_details[0]['shape']}")
print(f"Dtype: {output_details[0]['dtype']}")
