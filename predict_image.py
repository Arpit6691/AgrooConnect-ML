import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

MODEL_PATH = "plant_disease_model.keras"
CLASS_NAMES_PATH = "class_names.json"

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Load all 38 class names
with open(CLASS_NAMES_PATH, "r") as file:
    class_names = json.load(file)

# Check image path
if len(sys.argv) < 2:
    print("Usage: python predict_image.py <image_path>")
    sys.exit()

image_path = sys.argv[1]

# Load image
img = image.load_img(
    image_path,
    target_size=(224, 224)
)

# Convert image to array
img_array = image.img_to_array(img)

# Add batch dimension
img_array = np.expand_dims(img_array, axis=0)

# IMPORTANT:
# Do NOT use MobileNetV2 preprocess_input here.
# The model already contains:
# layers.Rescaling(1./127.5, offset=-1)

# Predict
predictions = model.predict(img_array, verbose=0)

predicted_index = np.argmax(predictions[0])
confidence = float(np.max(predictions[0]) * 100)

predicted_class = class_names[predicted_index]

# Separate crop and disease
parts = predicted_class.split("___")

crop = parts[0].replace("_", " ").replace(",", "")

condition = parts[1] if len(parts) > 1 else "Unknown"

disease = condition.replace("_", " ")

# Determine status
if disease.lower() == "healthy":
    status = "Healthy"
    disease = "Healthy"
else:
    status = "Diseased"

# Final result
result = {
    "crop": crop,
    "disease": disease,
    "status": status,
    "confidence": round(confidence, 2)
}

print(json.dumps(result, indent=2))