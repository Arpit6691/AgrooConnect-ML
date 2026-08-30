import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

MODEL_PATH = "plant_disease_model.keras"
CLASS_NAMES_PATH = "class_names.json"

print("Loading 39-class model...")

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load correct 39 class names
with open(CLASS_NAMES_PATH, "r") as file:
    class_names = json.load(file)

print("Model loaded successfully.")
print(f"Loaded {len(class_names)} classes.")

# Get image path from command line
if len(sys.argv) < 2:
    print("\nUsage:")
    print('python predict.py "image_name.jpg"')
    sys.exit(1)

IMAGE_PATH = sys.argv[1]

# Load image
img = image.load_img(
    IMAGE_PATH,
    target_size=(224, 224)
)

# Convert to array
img_array = image.img_to_array(img)

# Add batch dimension
img_array = np.expand_dims(
    img_array,
    axis=0
)

# Predict
predictions = model.predict(
    img_array,
    verbose=0
)

score = predictions[0]

predicted_index = int(np.argmax(score))
confidence = float(score[predicted_index] * 100)

# Safety check
if predicted_index >= len(class_names):
    raise ValueError(
        f"Model predicted index {predicted_index}, "
        f"but only {len(class_names)} class names exist."
    )

predicted_class = class_names[predicted_index]

print("\n========== RESULT ==========")
print("Prediction:", predicted_class)
print(f"Confidence: {confidence:.2f}%")
print("============================\n")