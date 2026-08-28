import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# -----------------------------
# LOAD TRAINED MODEL
# -----------------------------

model = tf.keras.models.load_model(
    "plant_disease_model.keras"
)

# -----------------------------
# CLASS NAMES
# IMPORTANT: Must be in the
# same order as training
# -----------------------------

class_names = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy"
]

# -----------------------------
# IMAGE PATH
# -----------------------------

IMAGE_PATH = "test_leaf.jpg"

# -----------------------------
# LOAD AND PREPROCESS IMAGE
# -----------------------------

img = image.load_img(
    IMAGE_PATH,
    target_size=(224, 224)
)

img_array = image.img_to_array(img)

img_array = np.expand_dims(
    img_array,
    axis=0
)

# -----------------------------
# MAKE PREDICTION
# -----------------------------

predictions = model.predict(img_array)

score = predictions[0]

predicted_index = np.argmax(score)

predicted_class = class_names[predicted_index]

confidence = float(
    score[predicted_index] * 100
)

# -----------------------------
# SHOW RESULT
# -----------------------------

print("\n========== RESULT ==========")

print("Prediction:", predicted_class)

print(f"Confidence: {confidence:.2f}%")

print("============================\n")