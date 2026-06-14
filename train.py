"""
CropGuard AI - Training Script
================================
Uses MobileNetV2 transfer learning on the PlantVillage dataset.

SETUP:
  1. Download dataset from Kaggle:
       kaggle datasets download -d abdallahalidev/plantvillage-dataset
     OR download manually from:
       https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
  2. Unzip so your folder structure looks like:
       PlantVillage/
           Apple___Apple_scab/
           Apple___healthy/
           Tomato___Bacterial_spot/
           ... (38 folders total)
  3. Run: python train.py

OUTPUT:
  - crop_disease_model.h5   (trained model)
  - class_indices.json      (label mapping for the app)
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# ─────────────────────────────
# Config
# ─────────────────────────────
DATA_DIR      = "PlantVillage"
IMG_SIZE      = 224
BATCH_SIZE    = 32
EPOCHS_FROZEN = 8      # Phase 1: train head only
EPOCHS_FINE   = 10     # Phase 2: fine-tune top layers
MODEL_PATH    = "crop_disease_model.h5"
INDICES_PATH  = "class_indices.json"


# ─────────────────────────────
# Data generators
# ─────────────────────────────
train_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    zoom_range=0.15,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    validation_split=0.2,
)

val_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
)

train_data = train_gen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
)

val_data = val_gen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
)

num_classes = len(train_data.class_indices)
print(f"[INFO] Found {num_classes} disease classes.")
print(f"[INFO] Training samples : {train_data.samples}")
print(f"[INFO] Validation samples: {val_data.samples}")

# Save class index mapping  (used by app.py)
class_indices = {v: k for k, v in train_data.class_indices.items()}
with open(INDICES_PATH, "w") as f:
    json.dump(class_indices, f)
print(f"[INFO] Class indices saved to {INDICES_PATH}")


# ─────────────────────────────
# Build model
# ─────────────────────────────
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)
base_model.trainable = False   # Freeze all base layers initially

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(num_classes, activation="softmax"),
], name="CropGuardAI")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()


# ─────────────────────────────
# Callbacks
# ─────────────────────────────
callbacks_phase1 = [
    ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_accuracy", verbose=1),
    EarlyStopping(patience=4, restore_best_weights=True, monitor="val_accuracy"),
]

callbacks_phase2 = [
    ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_accuracy", verbose=1),
    EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy"),
    ReduceLROnPlateau(factor=0.5, patience=3, monitor="val_loss", verbose=1),
]


# ─────────────────────────────
# Phase 1 — Train head only
# ─────────────────────────────
print("\n[PHASE 1] Training classifier head (base frozen)...")
history1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_FROZEN,
    callbacks=callbacks_phase1,
)


# ─────────────────────────────
# Phase 2 — Fine-tune top layers
# ─────────────────────────────
print("\n[PHASE 2] Fine-tuning top 30 layers of MobileNetV2...")
base_model.trainable = True
# Freeze all but the last 30 layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_FINE,
    callbacks=callbacks_phase2,
)


# ─────────────────────────────
# Final evaluation
# ─────────────────────────────
val_loss, val_acc = model.evaluate(val_data, verbose=0)
print(f"\n[RESULT] Final Validation Accuracy: {val_acc * 100:.2f}%")
print(f"[RESULT] Model saved to: {MODEL_PATH}")


# ─────────────────────────────
# Plot training curves
# ─────────────────────────────
acc  = history1.history["accuracy"]  + history2.history["accuracy"]
vacc = history1.history["val_accuracy"] + history2.history["val_accuracy"]
loss = history1.history["loss"]      + history2.history["loss"]
vloss= history1.history["val_loss"]  + history2.history["val_loss"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(acc, label="Train accuracy")
ax1.plot(vacc, label="Val accuracy")
ax1.axvline(EPOCHS_FROZEN, color="gray", linestyle="--", label="Fine-tune start")
ax1.set_title("Accuracy")
ax1.legend()

ax2.plot(loss, label="Train loss")
ax2.plot(vloss, label="Val loss")
ax2.axvline(EPOCHS_FROZEN, color="gray", linestyle="--", label="Fine-tune start")
ax2.set_title("Loss")
ax2.legend()

plt.tight_layout()
plt.savefig("training_curves.png", dpi=120)
print("[INFO] Training curves saved to training_curves.png")
plt.show()
