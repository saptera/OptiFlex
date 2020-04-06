import keras
from keras import regularizers
from keras.models import Model, Input
from keras.layers import Activation, BatchNormalization
from keras.layers import Conv2D, Conv2DTranspose
from keras.applications.resnet50 import ResNet50

"""Function list:
    deconv_resnet_model(img_width, img_height, num_joints, learning_rate): Standard version of FlexibleBaseline
    deconv_resnet_model_reduced(img_width, img_height, num_joints, learning_rate): Reduced version of FlexibleBaseline
    deconv_resnet_model_small(img_width, img_height, num_joints, learning_rate): Small version of FlexibleBaseline
"""


def deconv_resnet_model(img_width, img_height, num_joints, learning_rate):
    """ Standard version of FlexibleBaseline model

    Args:
        img_width (int): Width of the input image in number of pixels
        img_height (int): Height of the input image in number of pixels
        num_joints (int): number of joints of the dataset animal
        learning_rate (float): scalar value indicating speed of gradient update

    Returns:
        Keras model: untrained standard FlexibleBaseline model
    """
    input_img = Input(shape=(img_height, img_width, 3))
    resnet = ResNet50(weights="imagenet", input_tensor=input_img, include_top=False)

    res_conv3 = resnet.get_layer("activation_22").output
    res_conv3 = Conv2DTranspose(num_joints, (13, 13),
                                strides=(8, 8),
                                padding='same')(res_conv3)

    res_conv4 = resnet.get_layer("activation_40").output
    deconv1 = Conv2DTranspose(64, (13, 13),
                              strides=(4, 4),
                              padding='same',
                              kernel_initializer="glorot_normal",
                              kernel_regularizer=regularizers.l2(0.01))(res_conv4)
    bn1 = BatchNormalization()(deconv1)
    relu1 = Activation('relu')(bn1)

    deconv2 = Conv2DTranspose(64, (13, 13),
                              strides=(2, 2),
                              padding='same',
                              kernel_initializer="glorot_normal",
                              kernel_regularizer=regularizers.l2(0.01))(relu1)
    bn2 = BatchNormalization()(deconv2)
    relu2 = Activation('relu')(bn2)

    deconv3 = Conv2DTranspose(num_joints*2, (13, 13),
                              strides=(2, 2),
                              padding='same',
                              kernel_initializer="glorot_normal",
                              kernel_regularizer=regularizers.l2(0.01))(relu2)
    bn3 = BatchNormalization()(deconv3)
    relu3 = Activation('relu')(bn3)

    output = Conv2D(num_joints, (1, 1),
                    activation='linear',
                    padding='same',
                    strides=(1, 1),
                    kernel_regularizer=regularizers.l2(0.01))(relu3)

    model = Model(inputs=input_img, outputs=[res_conv3, output])

    adam = keras.optimizers.Adam(lr=learning_rate,
                                 beta_1=0.9,
                                 beta_2=0.999,
                                 epsilon=None,
                                 decay=0.0,
                                 amsgrad=True)

    model.compile(optimizer=adam,
                  loss="mean_squared_error",
                  metrics=['mae'])

    model.summary()
    return model


def deconv_resnet_model_reduced(img_width, img_height, num_joints, learning_rate):
    """ Reduced version of FlexibleBaseline model

    Args:
        img_width (int): Width of the input image in number of pixels
        img_height (int): Height of the input image in number of pixels
        num_joints (int): number of joints of the dataset animal
        learning_rate (float): scalar value indicating speed of gradient update

    Returns:
        Keras model: untrained reduced FlexibleBaseline model
    """
    input_img = Input(shape=(img_height, img_width, 3))
    resnet = ResNet50(weights="imagenet", input_tensor=input_img, include_top=False)

    res_conv2 = resnet.get_layer("activation_10").output
    res_conv2 = Conv2DTranspose(num_joints, (13, 13),
                                strides=(4, 4),
                                padding='same')(res_conv2)

    res_conv3 = resnet.get_layer("activation_22").output
    deconv1 = Conv2DTranspose(64, (13, 13),
                              strides=(2, 2),
                              padding='same',
                              kernel_initializer="glorot_normal",
                              kernel_regularizer=regularizers.l2(0.01))(res_conv3)
    bn1 = BatchNormalization()(deconv1)
    relu1 = Activation('relu')(bn1)

    deconv2 = Conv2DTranspose(64, (13, 13),
                              strides=(2, 2),
                              padding='same',
                              kernel_initializer="glorot_normal",
                              kernel_regularizer=regularizers.l2(0.01))(relu1)
    bn2 = BatchNormalization()(deconv2)
    relu2 = Activation('relu')(bn2)

    deconv3 = Conv2DTranspose(num_joints*2, (13, 13),
                              strides=(2, 2),
                              padding='same',
                              kernel_initializer="glorot_normal",
                              kernel_regularizer=regularizers.l2(0.01))(relu2)
    bn3 = BatchNormalization()(deconv3)
    relu3 = Activation('relu')(bn3)

    output = Conv2D(num_joints, (1, 1),
                    activation='linear',
                    padding='same',
                    strides=(1, 1),
                    kernel_regularizer=regularizers.l2(0.01))(relu3)

    model = Model(inputs=input_img, outputs=[res_conv2, output])

    adam = keras.optimizers.Adam(lr=learning_rate,
                                 beta_1=0.9,
                                 beta_2=0.999,
                                 epsilon=None,
                                 decay=0.0,
                                 amsgrad=True)

    model.compile(optimizer=adam,
                  loss="mean_squared_error",
                  metrics=['mae'])

    model.summary()
    return model


def deconv_resnet_model_small(img_width, img_height, num_joints, learning_rate):
    """ Small version of FlexibleBaseline model

    Args:
        img_width (int): Width of the input image in number of pixels
        img_height (int): Height of the input image in number of pixels
        num_joints (int): number of joints of the dataset animal
        learning_rate (float): scalar value indicating speed of gradient update

    Returns:
        Keras model: untrained small FlexibleBaseline model
    """
    input_img = Input(shape=(img_height, img_width, 3))
    resnet = ResNet50(weights="imagenet", input_tensor=input_img, include_top=False)

    res_conv2 = resnet.get_layer("activation_10").output
    deconv1 = Conv2DTranspose(32, (13, 13),
                              strides=(2, 2),
                              padding='same',
                              kernel_initializer="glorot_normal",
                              kernel_regularizer=regularizers.l2(0.01))(res_conv2)
    bn1 = BatchNormalization()(deconv1)
    relu1 = Activation('relu')(bn1)

    deconv2 = Conv2DTranspose(32, (13, 13),
                              strides=(2, 2),
                              padding='same',
                              kernel_initializer="glorot_normal",
                              kernel_regularizer=regularizers.l2(0.01))(relu1)
    bn2 = BatchNormalization()(deconv2)
    relu2 = Activation('relu')(bn2)

    deconv3 = Conv2DTranspose(num_joints*2, (13, 13),
                              strides=(1, 1),
                              padding='same',
                              kernel_initializer="glorot_normal",
                              kernel_regularizer=regularizers.l2(0.01))(relu2)
    bn3 = BatchNormalization()(deconv3)
    relu3 = Activation('relu')(bn3)

    output = Conv2D(num_joints, (1, 1),
                    activation='linear',
                    padding='same',
                    strides=(1, 1),
                    kernel_regularizer=regularizers.l2(0.01))(relu3)

    model = Model(inputs=input_img, outputs=output)

    adam = keras.optimizers.Adam(lr=learning_rate,
                                 beta_1=0.9,
                                 beta_2=0.999,
                                 epsilon=None,
                                 decay=0.0,
                                 amsgrad=True)

    model.compile(optimizer=adam,
                  loss="mean_squared_error",
                  metrics=['mae'])

    model.summary()
    return model
