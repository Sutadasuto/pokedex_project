import tensorflow as tf

# More stable on TF 2.10 Windows
tf.compat.v1.disable_eager_execution()

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0

# =========================================================
# CONFIG
# =========================================================

DATASET_DIR = r"./pokedex_data"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

INITIAL_EPOCHS = 50
FINE_TUNE_EPOCHS = 50

INITIAL_LR = 1e-3
FINE_TUNE_LR = 1e-5

WEIGHTS_OUTPUT = "pokedex_weights.h5"

# =========================================================
# LOAD DATASET
# =========================================================

train_ds = keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.5,
    subset="training",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

val_ds = keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.5,
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("Classes:")

for i, name in enumerate(class_names):
    print(i, name)

# =========================================================
# PERFORMANCE
# =========================================================

AUTOTUNE = tf.data.AUTOTUNE

# =========================================================
# PREPROCESSING / AUGMENTATION
# =========================================================

def preprocess_train(image, label):

    # Normalize
    image = tf.cast(image, tf.float32) / 255.0

    # Data augmentation
    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_brightness(
        image,
        max_delta=0.1
    )

    image = tf.image.random_contrast(
        image,
        lower=0.9,
        upper=1.1
    )

    # Clamp after augmentations
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label


def preprocess_val(image, label):

    image = tf.cast(image, tf.float32) / 255.0

    return image, label

# =========================================================
# APPLY DATA PIPELINE
# =========================================================

train_ds = train_ds.map(
    preprocess_train,
    num_parallel_calls=AUTOTUNE
)

val_ds = val_ds.map(
    preprocess_val,
    num_parallel_calls=AUTOTUNE
)

train_ds = train_ds.prefetch(AUTOTUNE)

val_ds = val_ds.prefetch(AUTOTUNE)

# =========================================================
# MODEL
# =========================================================

inputs = keras.Input(shape=(224, 224, 3))

base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_tensor=inputs
)

base_model.trainable = False

x = layers.GlobalAveragePooling2D()(base_model.output)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    num_classes,
    activation="softmax"
)(x)

model = keras.Model(inputs, outputs)

# =========================================================
# COMPILE
# =========================================================

model.compile(
    optimizer=keras.optimizers.Adam(INITIAL_LR),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================================================
# CALLBACKS
# =========================================================

callbacks = [

    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=50,
        restore_best_weights=True
    )

]

# =========================================================
# STAGE 1 TRAINING
# =========================================================

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS,
    callbacks=callbacks
)

# =========================================================
# STAGE 2 FINE TUNING
# =========================================================

base_model.trainable = True

fine_tune_at = 100

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(FINE_TUNE_LR),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
    initial_epoch=history.epoch[-1],
    callbacks=callbacks
)

# =========================================================
# SAVE WEIGHTS
# =========================================================

model.save_weights(WEIGHTS_OUTPUT)

# =========================================================
# SAVE LABELS
# =========================================================

with open("labels.txt", "w", encoding="utf-8") as f:

    for name in class_names:
        f.write(name + "\n")

print("Training complete.")