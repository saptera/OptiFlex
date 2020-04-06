import keras
from keras.models import Model, Input
from keras.layers import Conv3D


def flow_model(img_width, img_height, num_joints, learning_rate, frame_range):
    """ 3D convolution part of OpticalFlow model (previous parts completed in data preparation process)

        Args:
            img_width (int): Width of the input image in number of pixels
            img_height (int): Height of the input image in number of pixels
            num_joints (int): number of joints of the dataset animal
            learning_rate (float): scalar value indicating speed of gradient update
            frame_range (int): Number of adjacent frames to consider from before and after each reference frame

        Returns:
            Keras model: untrained 3D convolution layer of OpticalFlow model
    """

    num_frame = 2*frame_range + 1
    warped_hms = Input(shape=(num_frame, img_height, img_width, num_joints))

    hm_preds = Conv3D(1, (1, 1, num_frame), data_format="channels_first",  padding='same')(warped_hms)

    model = Model(inputs=warped_hms, outputs=hm_preds)

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
