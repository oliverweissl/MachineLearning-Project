from tensorflow import keras
from tensorflow.keras import layers

def create_model(input_shape):
    IMG_SIZE = input_shape[0]
    inputs = keras.Input(shape=input_shape)
    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomRotation(0.1),
        ]
    )
    x = data_augmentation(inputs)
    x = layers.Flatten()(x)
    x = layers.Dense(IMG_SIZE*6, activation="relu")(x)
    x = layers.Dense(IMG_SIZE*4, activation="relu")(x)
    x = layers.Dense(IMG_SIZE*2, activation="relu")(x)
    x = layers.Dense(IMG_SIZE, activation="relu")(x)

    outputs = layers.Dense(units=26, activation="softmax")(x)
    return keras.Model(inputs, outputs)