from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2

def create_model(input_shape, num_classes):
    inputs = keras.Input(shape=(input_shape[1], input_shape[0], input_shape[2]))

    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomRotation(0.1),
        ]
    )
    x = data_augmentation(inputs)

    x = layers.Conv2D(16, 3, strides=2, padding="same",activation = "swish", kernel_regularizer = l2(1e-4))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(32, 3, padding="same",activation = "swish", kernel_regularizer = l2(1e-4))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(32, 3, padding="same",activation = "swish", kernel_regularizer = l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)

    previous_block_activation = x

    for size in [64]:
        x = layers.SeparableConv2D(size, 3, padding="same",activation = "swish", kernel_regularizer = l2(1e-4))(x)
        x = layers.BatchNormalization()(x)

        x = layers.SeparableConv2D(size, 3, padding="same",activation = "swish", kernel_regularizer = l2(1e-4))(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same", kernel_regularizer = l2(1e-4))(previous_block_activation)
        x = layers.add([x, residual])

        x = layers.Dropout(0.1)(x)
        previous_block_activation = x

    x = layers.SeparableConv2D(128, 3, padding="same",activation = "swish", kernel_regularizer = l2(1e-4))(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
    x = layers.Flatten()(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)