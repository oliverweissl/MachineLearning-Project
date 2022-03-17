from tensorflow import keras
from tensorflow.keras import layers

def create_model(input_shape, num_classes):

    inputs = keras.Input(shape=input_shape)
    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomRotation(factor=0.20),
        ]
    )
    x = data_augmentation(inputs)
    x = layers.Conv2D(filters=64, kernel_size=5, activation="relu", padding='same', input_shape=[64, 64, 1])(x)
    x = layers.Dropout(0.3)(x)
    x = layers.MaxPool2D()(x)

    x = layers.Conv2D(filters=128, kernel_size=3, activation="relu", padding='same', input_shape=[64, 64, 1])(x)
    x = layers.Dropout(0.3)(x)
    x = layers.MaxPool2D()(x)

    x = layers.Conv2D(filters=256, kernel_size=3, activation="relu", padding='same', input_shape=[64, 64, 1])(x)
    x = layers.Dropout(0.3)(x)
    x = layers.MaxPool2D()(x)
    
    x = layers.Conv2D(filters=512, kernel_size=3, activation="relu", padding='same', input_shape=[64, 64, 1])(x)
    x = layers.Dropout(0.3)(x)
    x = layers.MaxPool2D()(x)


    x =  layers.Flatten()(x)
    x = layers.Dense(units=256, activation="relu")(x)

    outputs = layers.Dense(units=num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)