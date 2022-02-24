from tensorflow import keras
from tensorflow.keras import layers

def create_model(input_shape):

    inputs = keras.Input(shape=input_shape)
    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomRotation(0.1),
        ]
    )
    x = data_augmentation(inputs)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)

    outputs = layers.Dense(units=26, activation="softmax")(x)
    return keras.Model(inputs, outputs)