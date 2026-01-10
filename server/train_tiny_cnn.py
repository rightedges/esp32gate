import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

# === Load ROI from JSON ===
def load_roi(filepath="roi.json"):
    with open(filepath, "r") as f:
        roi_data = json.load(f)
    return roi_data["x"], roi_data["y"], roi_data["x1"], roi_data["y1"]

ROI = load_roi()

def crop_and_preprocess(img_path, target_size=(48, 48)):
    img = Image.open(img_path).convert('L') # Convert to Grayscale
    x, y, x1, y1 = ROI
    img = img.crop((x, y, x1, y1))
    # Use Image.BOX to match ESP32 averaging logic
    resample_filter = Image.BOX if hasattr(Image, 'BOX') else Image.BILINEAR
    img = img.resize(target_size, resample=resample_filter)
    return np.array(img).astype('float32') / 255.0

def load_dataset(dataset_path):
    x_data = []
    y_labels = []
    classes = ['closed', 'open']
    
    for label, class_name in enumerate(classes):
        # The data is in data/train/closed and data/train/open
        class_dir = os.path.join(dataset_path, 'train', class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: {class_dir} not found")
            continue
        for img_name in os.listdir(class_dir):
            if img_name.endswith('.jpg'):
                img_path = os.path.join(class_dir, img_name)
                img_array = crop_and_preprocess(img_path)
                x_data.append(img_array.reshape(48, 48, 1))
                y_labels.append(label)
                
    return np.array(x_data), np.array(y_labels)

# Load and preprocess data
DATA_DIR = "./data"
X, y = load_dataset(DATA_DIR)
print(f"Loaded {len(X)} images.")

# SHUFFLE DATA (Crucial for subset='training'/'validation')
indices = np.arange(len(X))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# Data Augmentation logic
datagen = ImageDataGenerator(
    brightness_range=[0.7, 1.3],
    zoom_range=0.1,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=False,
    validation_split=0.2
)

# Tiny CNN Architecture (Strided Conv for TFLM Compatibility)
model = models.Sequential([
    layers.Input(shape=(48, 48, 1)),
    layers.Conv2D(16, (3, 3), strides=(2, 2), activation='relu', padding='same'),
    layers.Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same'),
    layers.Flatten(),
    layers.Dense(16, activation='relu'),
    layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Starting training...")
# Simple training without generator for faster convergence on small data
history = model.fit(X, y, epochs=100, validation_split=0.2)

# Export model in Keras format
model.save("gate_detector_tiny.keras")
print("Tiny CNN model saved as gate_detector_tiny.keras")

# Allow history to be accessed
acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]
loss = history.history['loss'][-1]
val_loss = history.history['val_loss'][-1]

print("\n" + "="*30)
print("     TRAINING RESULTS")
print("="*30)
print(f"Final Accuracy:      {acc*100:.2f}%")
print(f"Final Val Accuracy:  {val_acc*100:.2f}%")
print(f"Final Loss:          {loss:.4f}")
print(f"Final Val Loss:      {val_loss:.4f}")
print("="*30 + "\n")
