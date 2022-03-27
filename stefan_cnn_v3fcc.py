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
    CONV_UNITS_BODY = [64, 128]#, 256, 512, 512]


    x = data_augmentation(inputs)

    # Entry block
    #version tf 2.4.1
    #x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    #-----------------
    x = layers.Conv2D(16, 3, strides=2, padding="same", kernel_regularizer = l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)

    x = layers.Conv2D(32, 3, padding="same", kernel_regularizer = l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)


    x = layers.Dropout(0.1)(x)

    previous_block_activation = x  # Set aside residualcre

    for size in CONV_UNITS_BODY:
        x = layers.Activation("swish")(x)
        x = layers.SeparableConv2D(size, 3, padding="same", kernel_regularizer = l2(1e-4))(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("swish")(x)
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

    x = layers.SeparableConv2D(128, 3, padding="same", kernel_regularizer = l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
    x = layers.Flatten()(x)


    x = layers.Dense(1024, activation="swish", kernel_regularizer = l2(1e-4))(x)


    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation= "softmax")(x)
    return keras.Model(inputs, outputs)