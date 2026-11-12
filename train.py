# KHILONA - Plastic object colour classifier (CNN built from scratch)
# Classes: Blue -> Conveyor Belt A, Yellow -> Conveyor Belt B, Purple -> Conveyor Belt C

import tensorflow as tf
from tensorflow.keras import layers, models

# ---------- 1. Settings ----------
IMG_SIZE = (128, 128)      # every image is resized to 128 x 128
BATCH_SIZE = 16
EPOCHS = 10

# ---------- 2. Load our own dataset ----------
# dataset/
#   train/  A_blue, B_yellow, C_purple   (folder name = class label)
#   test/   A_blue, B_yellow, C_purple   (images the model never sees in training)
train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/train",
    validation_split=0.2,      # 20% of train images are kept for validation
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/train",
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/test",
    shuffle=False,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

print("Classes:", train_ds.class_names)   # ['A_blue', 'B_yellow', 'C_purple']

# ---------- 3. Build the CNN (no pretrained weights) ----------
model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Rescaling(1.0 / 255),                      # 0-255 -> 0-1

    layers.Conv2D(16, (3, 3), activation="relu"),     # block 1
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(32, (3, 3), activation="relu"),     # block 2
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),     # block 3
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(3, activation="softmax"),            # 3 classes: A, B, C
])

model.summary()

# ---------- 4. Compile and train ----------
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",           # labels are 0, 1, 2
    metrics=["accuracy"],
)

history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# ---------- 5. Test accuracy ----------
test_loss, test_acc = model.evaluate(test_ds)
print("Test Accuracy: {:.2f}%".format(test_acc * 100))

# ---------- 6. Save the trained model ----------
model.save("khilona_model.keras")
print("Model saved as khilona_model.keras")
