# train.py
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight  # For bias reduction
import numpy as np
import matplotlib.pyplot as plt
import json

# Paths to training and validation directories
train_dir = './data/train_final'
val_dir = './data/val'

# Set training parameters
img_height, img_width = 224, 224
batch_size = 16
epochs = 10  # Adjust based on performance
learning_rate = 1e-4

# Data generator for training (with augmentation)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Data generator for validation (only rescaling)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

# --- Compute class weights to reduce bias ---
# This ensures that if one class is underrepresented, its loss is weighted more.
classes = train_generator.classes  # Array of class indices for training images
class_labels = np.unique(classes)
class_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=classes)
# Convert the class weights array into a dictionary
class_weight_dict = dict(zip(class_labels, class_weights))
print("Computed class weights:", class_weight_dict)
# ---------------------------------------------------

# Load pre-trained EfficientNetB0 model without top layers and freeze it
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
base_model.trainable = False

# Add custom classification layers on top for binary classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Set callbacks for early stopping and checkpointing
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ModelCheckpoint('deepfake_model.h5', monitor='val_loss', save_best_only=True)
]

# Train the model, using class_weight to reduce bias
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    callbacks=callbacks,
    class_weight=class_weight_dict  # <-- This parameter balances the classes
)

print("Training complete!")

# Save the training history to a JSON file for plotting later
with open("training_history.json", "w") as f:
    json.dump(history.history, f)

# Plot training and validation accuracy and loss, then save the plot
epochs_range = range(1, len(history.history["accuracy"]) + 1)

plt.figure(figsize=(12, 5))
# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, history.history["accuracy"], "bo-", label="Training Accuracy")
plt.plot(epochs_range, history.history["val_accuracy"], "ro-", label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
# Plot loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, history.history["loss"], "bo-", label="Training Loss")
plt.plot(epochs_range, history.history["val_loss"], "ro-", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("static/training_plots.png")
plt.show()
