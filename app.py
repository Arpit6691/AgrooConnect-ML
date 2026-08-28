import os
import json
from io import BytesIO

import numpy as np
import tensorflow as tf

from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image


app = Flask(__name__)


# -----------------------------------
# PATHS
# -----------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "plant_disease_model.keras"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR,
    "class_names.json"
)


# -----------------------------------
# LOAD MODEL
# -----------------------------------

print("Loading plant disease model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# -----------------------------------
# LOAD CLASS NAMES
# -----------------------------------

with open(CLASS_NAMES_PATH, "r") as file:
    class_names = json.load(file)

print(f"Loaded {len(class_names)} classes.")


# -----------------------------------
# HEALTH CHECK
# -----------------------------------

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "AgrooConnect ML API is running",
        "classes": len(class_names)
    })


# -----------------------------------
# PREDICTION API
# -----------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    # Check whether image exists
    if "image" not in request.files:
        return jsonify({
            "error": "No image file provided"
        }), 400

    file = request.files["image"]

    # Check whether file is selected
    if file.filename == "":
        return jsonify({
            "error": "No image selected"
        }), 400

    try:

        print("Received image:", file.filename)

        # -----------------------------------
        # CONVERT FLASK FILESTORAGE TO BYTES
        # -----------------------------------

        image_data = file.read()

        if not image_data:
            return jsonify({
                "error": "Uploaded image is empty"
            }), 400

        image_bytes = BytesIO(image_data)

        # -----------------------------------
        # LOAD IMAGE
        # -----------------------------------

        img = image.load_img(
            image_bytes,
            target_size=(224, 224)
        )

        # Convert image to NumPy array
        img_array = image.img_to_array(img)

        # Add batch dimension
        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        # -----------------------------------
        # MAKE PREDICTION
        # -----------------------------------

        predictions = model.predict(
            img_array,
            verbose=0
        )

        # Get predicted class index
        predicted_index = int(
            np.argmax(predictions[0])
        )

        # Get confidence percentage
        confidence = float(
            np.max(predictions[0]) * 100
        )

        # Get predicted class
        predicted_class = class_names[predicted_index]

        print(
            "Prediction:",
            predicted_class,
            "| Confidence:",
            confidence
        )

        # -----------------------------------
        # SEPARATE CROP AND DISEASE
        # -----------------------------------

        parts = predicted_class.split("___")

        crop = parts[0]

        crop = (
            crop
            .replace("_", " ")
            .replace(",", "")
        )

        condition = (
            parts[1]
            if len(parts) > 1
            else "Unknown"
        )

        disease = condition.replace(
            "_",
            " "
        )

        # -----------------------------------
        # DETERMINE STATUS
        # -----------------------------------

        if disease.lower() == "healthy":

            status = "Healthy"

            disease = "Healthy"

        else:

            status = "Diseased"

        # -----------------------------------
        # RETURN RESULT
        # -----------------------------------

        result = {
            "crop": crop,
            "disease": disease,
            "status": status,
            "confidence": round(
                confidence,
                2
            )
        }

        print("Returning result:", result)

        return jsonify(result)

    except Exception as error:

        print(
            "Prediction error:",
            str(error)
        )

        return jsonify({
            "error": str(error)
        }), 500


# -----------------------------------
# RUN SERVER
# -----------------------------------

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )