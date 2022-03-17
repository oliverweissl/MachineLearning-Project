import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_model(input_shape, num_classes):

    inputs = keras.Input(shape=input_shape)
    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomRotation(0.1),
        ]
    )
    x = data_augmentation(inputs)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation=tf.keras.layers.LeakyReLU(alpha=0.2))(x)
    x = layers.Dense(256, activation=tf.keras.layers.LeakyReLU(alpha=0.2))(x)
    x = layers.Dense(128, activation=tf.keras.layers.LeakyReLU(alpha=0.2))(x)
    x = layers.Dense(64, activation=tf.keras.layers.LeakyReLU(alpha=0.2))(x)

    outputs = layers.Dense(units=num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)