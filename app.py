import os

os.environ["OMP_NUM_THREADS"] = "1"

import json
from io import BytesIO

import numpy as np
from PIL import Image

from flask import Flask, request, jsonify

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter


app = Flask(__name__)


# -----------------------------------
# PATHS
# -----------------------------------

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "plant_disease_model.tflite"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR,
    "class_names.json"
)


# -----------------------------------
# LOAD TFLITE MODEL
# -----------------------------------

print("Loading TensorFlow Lite model...")

interpreter = Interpreter(
    model_path=MODEL_PATH,
    num_threads=1
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("TFLite model loaded successfully.")


# -----------------------------------
# LOAD CLASS NAMES
# -----------------------------------

with open(
    CLASS_NAMES_PATH,
    "r"
) as file:

    class_names = json.load(file)

print(
    f"Loaded {len(class_names)} classes."
)


# -----------------------------------
# HEALTH CHECK
# -----------------------------------

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message": "AgrooConnect ML TFLite API is running",
        "classes": len(class_names)
    })


# -----------------------------------
# PREDICTION API
# -----------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:

        return jsonify({
            "error": "No image file provided"
        }), 400

    file = request.files["image"]

    if file.filename == "":

        return jsonify({
            "error": "No image selected"
        }), 400

    try:

        print(
            "Received image:",
            file.filename
        )

        # Read image
        image_data = file.read()

        if not image_data:

            return jsonify({
                "error": "Uploaded image is empty"
            }), 400


        # Load image
        img = Image.open(
            BytesIO(image_data)
        ).convert("RGB")

        img = img.resize(
            (224, 224)
        )


        # Convert to NumPy
        img_array = np.array(
            img,
            dtype=np.float32
        )

        img_array = np.expand_dims(
            img_array,
            axis=0
        )


        # Run TFLite prediction
        input_index = input_details[0]["index"]
        output_index = output_details[0]["index"]

        interpreter.set_tensor(
            input_index,
            img_array
        )

        interpreter.invoke()

        predictions = interpreter.get_tensor(
            output_index
        )


        # Get predicted class
        predicted_index = int(
            np.argmax(
                predictions[0]
            )
        )

        confidence = float(
            np.max(
                predictions[0]
            ) * 100
        )

        predicted_class = class_names[
            predicted_index
        ]

        print(
            "Prediction:",
            predicted_class,
            "| Confidence:",
            confidence
        )


        # -----------------------------------
        # HANDLE INVALID IMAGE
        # -----------------------------------

        if predicted_class == "Other_Invalid":

            result = {
                "crop": "Other",
                "disease": "Invalid image",
                "status": "Invalid",
                "confidence": round(
                    confidence,
                    2
                )
            }


        # -----------------------------------
        # HANDLE PLANT IMAGE
        # -----------------------------------

        else:

            parts = predicted_class.split(
                "___"
            )

            crop = (
                parts[0]
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

            if disease.lower() == "healthy":

                status = "Healthy"
                disease = "Healthy"

            else:

                status = "Diseased"


            result = {
                "crop": crop,
                "disease": disease,
                "status": status,
                "confidence": round(
                    confidence,
                    2
                )
            }


        # -----------------------------------
        # RETURN RESULT
        # -----------------------------------

        print(
            "Returning result:",
            result
        )

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