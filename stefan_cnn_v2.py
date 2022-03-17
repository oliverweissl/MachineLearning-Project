from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2

def create_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    ACTIVATION_STR = "swish"

    data_augmentation = keras.Sequential(
        [
            # version tf 2.4.1:
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomRotation(0.1),
        ]
    )
    x = data_augmentation(inputs)

    x = layers.Conv2D(16, 3, strides=2, padding="same", kernel_regularizer = l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(ACTIVATION_STR)(x)

    x = layers.Conv2D(32, 3, padding="same", kernel_regularizer = l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(ACTIVATION_STR)(x)
    x = layers.Dropout(0.1)(x)

    previous_block_activation = x  # Set aside residual

    for size in [64]:
        x = layers.Activation(ACTIVATION_STR)(x)
        x = layers.SeparableConv2D(size, 3, padding="same", kernel_regularizer = l2(1e-4))(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation(ACTIVATION_STR)(x)
        x = layers.SeparableConv2D(size, 3, padding="same", kernel_regularizer = l2(1e-4))(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same", kernel_regularizer = l2(1e-4))(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        x = layers.Dropout(0.1)(x)
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(64, 3, padding="same", kernel_regularizer = l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(ACTIVATION_STR)(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)