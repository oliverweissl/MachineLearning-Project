import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_model(input_shape, num_classes):
    IMG_SIZE = input_shape[0]
    inputs = keras.Input(shape=input_shape)
    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomRotation(0.1),
        ]
    )

    KERNEL = (3, 3)

    x = data_augmentation(inputs)
    # layer 1
    x = layers.Conv2D(512, KERNEL, 1, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.2)(x)

    # layer 2
    x = layers.Conv2D(256, KERNEL, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.2)(x)

    # layer 3
    x = layers.Conv2D(128, KERNEL, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.2)(x)

    # layer 4
    x = layers.Conv2D(64, KERNEL, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(units=IMG_SIZE)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(units=IMG_SIZE / 2)(x)

    outputs = layers.Dense(units=num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)