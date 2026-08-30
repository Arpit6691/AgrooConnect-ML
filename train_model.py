# import tensorflow as tf
# from tensorflow.keras.preprocessing import image_dataset_from_directory
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras import layers, models

# # -----------------------------
# # SETTINGS
# # -----------------------------

# DATASET_PATH = "dataset"
# IMAGE_SIZE = (224, 224)
# BATCH_SIZE = 32
# EPOCHS = 5

# # -----------------------------
# # LOAD DATASET
# # -----------------------------

# train_dataset = image_dataset_from_directory(
#     DATASET_PATH,
#     validation_split=0.2,
#     subset="training",
#     seed=123,
#     image_size=IMAGE_SIZE,
#     batch_size=BATCH_SIZE
# )

# validation_dataset = image_dataset_from_directory(
#     DATASET_PATH,
#     validation_split=0.2,
#     subset="validation",
#     seed=123,
#     image_size=IMAGE_SIZE,
#     batch_size=BATCH_SIZE
# )

# # Get class names
# class_names = train_dataset.class_names

# print("\nClasses:")
# print(class_names)

# # -----------------------------
# # OPTIMIZE DATA LOADING
# # -----------------------------

# AUTOTUNE = tf.data.AUTOTUNE

# train_dataset = train_dataset.prefetch(
#     buffer_size=AUTOTUNE
# )

# validation_dataset = validation_dataset.prefetch(
#     buffer_size=AUTOTUNE
# )

# # -----------------------------
# # LOAD MOBILENETV2
# # -----------------------------

# base_model = MobileNetV2(
#     input_shape=(224, 224, 3),
#     include_top=False,
#     weights="imagenet"
# )

# # Freeze pre-trained layers
# base_model.trainable = False

# # -----------------------------
# # BUILD MODEL
# # -----------------------------

# model = models.Sequential([
#     layers.Rescaling(1./127.5, offset=-1),
    
#     base_model,
    
#     layers.GlobalAveragePooling2D(),
    
#     layers.Dropout(0.2),
    
#     layers.Dense(
#         len(class_names),
#         activation="softmax"
#     )
# ])

# # -----------------------------
# # COMPILE MODEL
# # -----------------------------

# model.compile(
#     optimizer="adam",
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"]
# )

# # -----------------------------
# # TRAIN MODEL
# # -----------------------------

# history = model.fit(
#     train_dataset,
#     validation_data=validation_dataset,
#     epochs=EPOCHS
# )

# # -----------------------------
# # SAVE MODEL
# # -----------------------------

# model.save("plant_disease_model.keras")

# print("\nModel training completed!")
# print("Model saved as: plant_disease_model.keras")



import os
import json
import tensorflow as tf

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

# -----------------------------
# SETTINGS
# -----------------------------

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset"
)

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32

EPOCHS = 10

MODEL_PATH = os.path.join(
    BASE_DIR,
    "plant_disease_model.keras"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR,
    "class_names.json"
)

# -----------------------------
# LOAD DATASET
# -----------------------------

train_dataset = image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

validation_dataset = image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# -----------------------------
# GET CLASS NAMES
# -----------------------------

class_names = train_dataset.class_names

print("\nNumber of classes:")
print(len(class_names))

print("\nClasses:")
for index, class_name in enumerate(class_names):
    print(f"{index}: {class_name}")

# -----------------------------
# VERIFY OTHER_INVALID EXISTS
# -----------------------------

if "Other_Invalid" not in class_names:
    raise ValueError(
        "Other_Invalid folder was not detected in dataset!"
    )

# -----------------------------
# SAVE CLASS NAMES
# -----------------------------

with open(
    CLASS_NAMES_PATH,
    "w"
) as file:
    json.dump(
        class_names,
        file,
        indent=4
    )

print(
    "\nNew class_names.json saved successfully."
)

# -----------------------------
# OPTIMIZE DATA LOADING
# -----------------------------

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
)

# -----------------------------
# LOAD MOBILENETV2
# -----------------------------

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pre-trained layers
base_model.trainable = False

# -----------------------------
# BUILD MODEL
# -----------------------------

model = models.Sequential([
    layers.Rescaling(
        1. / 127.5,
        offset=-1
    ),

    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dropout(
        0.2
    ),

    layers.Dense(
        len(class_names),
        activation="softmax"
    )
])

# -----------------------------
# COMPILE MODEL
# -----------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------
# TRAIN MODEL
# -----------------------------

print("\nStarting training...")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)

# -----------------------------
# SAVE MODEL
# -----------------------------

model.save(
    MODEL_PATH
)

print("\nModel training completed!")
print(
    "Model saved as:",
    MODEL_PATH
)

print(
    "Class names saved as:",
    CLASS_NAMES_PATH
)

print(
    "\nTotal trained classes:",
    len(class_names)
)