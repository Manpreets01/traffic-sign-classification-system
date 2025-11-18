#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: notebook.ipynb
Conversion Date: 2025-11-18T03:20:00.412Z
"""

!pip install -q tensorflow gradio

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr
import os


label_map = {
    "0": "Speed Limit",
    "1": "Stop",
    "2": "Turn left",
    "3": "Turn right",
    "4": "no entry towards left",
    "5": "no entry towards right",
    "6": "no entry",
    "7": "speed limit 40",
    "8": "speed limit 30",
    "9": "towards right",
    "10": "Prohibited area",
    "11": "no less than 30",
    "12": "towards left"
}


from google.colab import files
import zipfile, os

zip_path = "/content/Traffic-Sign-Detection-master.zip"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("/content/traffic_sign_data")

print("Files extracted to:", os.listdir("/content/traffic_sign_data"))


data_dir = "/content/traffic_sign_data/Traffic-Sign-Detection-master/dataset"

IMG_SIZE = 64
BATCH_SIZE = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    validation_split=0.2,  # 80% training, 20% validation
    subset="training",
    seed=123
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=123
)

class_names = train_ds.class_names
print("Detected Classes:", class_names)


def normalize_img(img, label):
    return tf.cast(img, tf.float32) / 255.0, label

train_ds = train_ds.map(normalize_img)
val_ds = val_ds.map(normalize_img)

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


EPOCHS = 10

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)


plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.legend()
plt.title("Model Accuracy")

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.legend()
plt.title("Model Loss")
plt.show()


import random
from PIL import Image

# Pick a random image path
random_class = random.choice(class_names)
class_path = os.path.join(data_dir, random_class)
img_path = os.path.join(class_path, random.choice(os.listdir(class_path)))

# Load and predict
img = Image.open(img_path).resize((IMG_SIZE, IMG_SIZE))
img_arr = np.expand_dims(np.array(img)/255.0, axis=0)
pred = model.predict(img_arr)
predicted_class = class_names[np.argmax(pred)]

plt.imshow(img)
plt.title(f"Predicted: {predicted_class}")
plt.axis("off")
plt.show()


def predict(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.expand_dims(np.array(img)/255.0, axis=0)
    pred = model.predict(img)
    class_idx = np.argmax(pred)
    class_label = class_names[class_idx]

    # Use our label_map to convert number → name
    readable_label = label_map.get(class_label, f"Class {class_label}")
    confidence = float(np.max(pred))
    return {readable_label: confidence}



demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Traffic Sign Recognition",
    description="Upload a traffic sign image to predict its class name."
)

demo.launch(share=True)