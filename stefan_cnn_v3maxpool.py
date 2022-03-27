from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2

def create_model(input_shape, num_classes):
    # Note: input is flipped to (height, width) instead of (width, height)
    inputs = keras.Input(shape=(input_shape[1], input_shape[0], input_shape[2]))

    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomRotation(0.1),
        ]
    )
    ACTIVATION_STR = "swish"
    FIRST_CONV_UNITS = 32
    SECOND_CONV_UNITS = 64
    CONV_UNITS_BODY = [128, 256, 512, 768]
    LAST_LAYER_UNITS = 1024
    LAST_LAYER_DROPOUT = 0.5
    x = data_augmentation(inputs)

    x = layers.Conv2D(FIRST_CONV_UNITS, 3, strides=2, padding="same", kernel_regularizer = l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(ACTIVATION_STR)(x)

    x = layers.Conv2D(SECOND_CONV_UNITS, 3, padding="same", kernel_regularizer = l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(ACTIVATION_STR)(x)
    x = layers.Dropout(0.1)(x)

    previous_block_activation = x  # Set aside residual

    for size in CONV_UNITS_BODY:
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

    x = layers.SeparableConv2D(LAST_LAYER_UNITS, 3, padding="same", kernel_regularizer = l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(ACTIVATION_STR)(x)
    x = layers.GlobalAveragePooling2D()(x)


    x = layers.Dropout(LAST_LAYER_DROPOUT)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)