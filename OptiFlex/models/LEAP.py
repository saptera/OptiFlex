import keras
from keras.models import Model, Input
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D


def LEAP(img_width, img_height, num_joints, learning_rate):
    """ LEAP model

        Args:
            img_width (int): Width of the input image in number of pixels
            img_height (int): Height of the input image in number of pixels
            num_joints (int): number of joints of the dataset animal
            learning_rate (float): scalar value indicating speed of gradient update

        Returns:
            Keras model: untrained LEAP model
        """

    input_tensor = Input(shape=(img_height, img_width, 3))

    x1 = Conv2D(64, kernel_size=3, padding="same", activation="relu")(input_tensor)
    x1 = Conv2D(64, kernel_size=3, padding="same", activation="relu")(x1)
    x1 = Conv2D(64, kernel_size=3, padding="same", activation="relu")(x1)
    x1_pool = MaxPooling2D(pool_size=2, strides=2, padding="same")(x1)

    x2 = Conv2D(128, kernel_size=3, padding="same", activation="relu")(x1_pool)
    x2 = Conv2D(128, kernel_size=3, padding="same", activation="relu")(x2)
    x2 = Conv2D(128, kernel_size=3, padding="same", activation="relu")(x2)
    x2_pool = MaxPooling2D(pool_size=2, strides=2, padding="same")(x2)

    x3 = Conv2D(256, kernel_size=3, padding="same", activation="relu")(x2_pool)
    x3 = Conv2D(256, kernel_size=3, padding="same", activation="relu")(x3)
    x3 = Conv2D(256, kernel_size=3, padding="same", activation="relu")(x3)

    x4 = Conv2DTranspose(128, kernel_size=3, strides=2, padding="same", activation="relu",
                         kernel_initializer="glorot_normal")(x3)
    x4 = Conv2D(128, kernel_size=3, padding="same", activation="relu")(x4)
    x4 = Conv2D(128, kernel_size=3, padding="same", activation="relu")(x4)

    output_tensor = Conv2DTranspose(num_joints, kernel_size=3, strides=2, padding="same", activation="linear",
                                    kernel_initializer="glorot_normal")(x4)

    model = Model(inputs=input_tensor, outputs=output_tensor)

    model.compile(optimizer=keras.optimizers.Adam(
        lr=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=None,
        decay=0.0,
        amsgrad=False),
        loss="mean_squared_error",
        metrics=['mae'])

    model.summary()
    return model

