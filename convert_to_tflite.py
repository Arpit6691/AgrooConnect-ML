import tensorflow as tf
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

keras_model_path = os.path.join(
    BASE_DIR,
    "plant_disease_model.keras"
)

tflite_model_path = os.path.join(
    BASE_DIR,
    "plant_disease_model.tflite"
)

print("Loading Keras model...")

model = tf.keras.models.load_model(
    keras_model_path
)

print("Converting model to TensorFlow Lite...")

converter = tf.lite.TFLiteConverter.from_keras_model(
    model
)

# Standard Float32 conversion
# This preserves the model's normal numerical precision.
tflite_model = converter.convert()

with open(
    tflite_model_path,
    "wb"
) as file:
    file.write(tflite_model)

print("TFLite model created successfully!")
print("Location:", tflite_model_path)

size_mb = os.path.getsize(
    tflite_model_path
) / (1024 * 1024)

print(f"File size: {size_mb:.2f} MB")
