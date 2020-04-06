import keras
from keras.models import Model, Input
from keras.layers import Conv2DTranspose
from keras.applications.resnet50 import ResNet50


def deeplabcut(img_width, img_height, num_joints, learning_rate):
    """ DeepLabCut model

    Args:
        img_width (int): Width of the input image in number of pixels
        img_height (int): Height of the input image in number of pixels
        num_joints (int): Number of joints of the dataset animal
        learning_rate (float): Scalar value indicating speed of gradient update

    Returns:
        Keras model: untrained DeepLabCut model
    """
    input_img = Input(shape=(img_height, img_width, 3))
    resnet = ResNet50(weights="imagenet", input_tensor=input_img, include_top=False)

    res_conv3 = resnet.get_layer("activation_22").output
    res_conv3 = Conv2DTranspose(num_joints, (11, 11),
                                strides=(8, 8),
                                padding='same')(res_conv3)
    res_conv5 = resnet.get_layer("activation_49").output

    deconv1 = Conv2DTranspose(num_joints, (36, 36),
                              strides=(32, 32),
                              padding='same')(res_conv5)
    
    model = Model(inputs=input_img, outputs=[res_conv3, deconv1])

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
