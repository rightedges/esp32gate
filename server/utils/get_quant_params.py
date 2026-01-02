import tensorflow as tf
import numpy as np

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="gate_detector_tiny.tflite")
interpreter.allocate_tensors()

# Get input and output details.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"Input Dtype: {input_details[0]['dtype']}")
print(f"Input Quantization: {input_details[0]['quantization']}")

scale, zero_point = input_details[0]['quantization']
print(f"SCALE: {scale}")
print(f"ZERO_POINT: {zero_point}")

# Also check output quantization
print(f"Output Dtype: {output_details[0]['dtype']}")
print(f"Output Quantization: {output_details[0]['quantization']}")
