# KHILONA - Predict the conveyor belt for one object image (no if / else used)

import numpy as np
import tensorflow as tf

IMG_SIZE = (128, 128)

# Same order as the folders: A_blue, B_yellow, C_purple
COLORS = ["Blue", "Yellow", "Purple"]
BELTS = ["Conveyor Belt A", "Conveyor Belt B", "Conveyor Belt C"]

model = tf.keras.models.load_model("khilona_model.keras")

path = input("Enter image path: ").strip().strip('"')

img = tf.keras.utils.load_img(path, target_size=IMG_SIZE)
img_array = tf.keras.utils.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)          # add batch dim

probabilities = model.predict(img_array)[0]            # 3 probabilities
index = int(np.argmax(probabilities))                  # highest probability

print("Detected colour :", COLORS[index])
print("Confidence      : {:.2f}%".format(probabilities[index] * 100))
print("Send to         :", BELTS[index])               # list lookup replaces if/else
