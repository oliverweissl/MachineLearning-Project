from tensorflow import keras
from tensorflow.keras import layers

def create_model(input_shape, num_classes):

    inputs = keras.Input(shape=input_shape)
    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomRotation(0.1),
        ]
    )
    kernel_size=(3,3)
    strides=(1,1)
    pool_size=(2,2)

    x = data_augmentation(inputs)
    x = layers.SeparableConv2D(50, kernel_size=kernel_size, strides=strides, padding='same', activation='relu')(x)
    x = layers.Conv2D(75, kernel_size=kernel_size, strides=strides, padding='same', activation='relu')(x)
    x = layers.MaxPool2D(pool_size=pool_size)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.SeparableConv2D(125, kernel_size=kernel_size, strides=strides, padding='same', activation='relu')(x)
    x = layers.MaxPool2D(pool_size=pool_size)(x)
    x = layers.Dropout(0.25)(x)

    #x = layers.SeparableConv2D(200, kernel_size=kernel_size, strides=strides, padding='same', activation='relu')(x)
    #x = layers.MaxPool2D(pool_size=pool_size)(x)
    #x = layers.Dropout(0.25)(x)

    x = layers.Flatten()(x)

    x = layers.Dense(500, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(250, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(125, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation="relu")(x)

    outputs = layers.Dense(units=num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)