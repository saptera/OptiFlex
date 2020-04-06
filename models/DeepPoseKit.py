# Refactored from https://github.com/jgraving/DeepPoseKit/blob/master/deepposekit/models

import keras
import numpy as np
from keras.models import Model, Input
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers import BatchNormalization, Concatenate

"""Function list:
# Building Blocks:
    num_transition(img_width, img_height): Calculating number of up or down transition for a given image dimension
    concatenate(inputs): Concatenation function that can handle input of length 1 properly
    dense_conv_2d(inputs, growth_rate=48, bottleneck_factor=1): Convolution layer with dense connections.
    dense_conv_block(inputs, growth_rate=48, n_layers=1, bottleneck_factor=1): Convolution block dense skip connections
    compression(inputs, compression_factor=0.5): Using 1x1 Conv2D filters to reduce number of model parameter
    transition_down(inputs, compression_factor=0.5, pool_size=2): Downsampling module with a max pooling layer
    transition_up(inputs, compression_factor=0.5): Upsampling module with a transpose convolution layer
    frontend(inputs, growth_rate=48, n_downsample=1, compression_factor=0.5, bottleneck_factor=1): Initial block in 
    StackedDenseNet for preliminary transformations before entering DenseNet modules.
    densenet(inputs, growth_rate=64, n_downsample=1, downsample_factor=0, compression_factor=0.5, bottleneck_factor=1): 
    Single DenseNet module with an Encoder and an Decoder segment.
    output_channels(inputs, num_joints): Output block applied to final and intermediate outputs
# Main Model:
    stacked_densenet(img_width, img_height, num_joints, learning_rate, num_stage, growth_rate=48, 
    bottleneck_factor=1, compression_factor=0.5): StackedDenseNet Model fully assembled
"""


# Building Blocks --------------------------------------------------------------------------------------------------- #

def num_transition(img_width, img_height):
    """ Calculating number of up or down transition for a given image dimension

    Args:
        img_width (int): Width of the input image in number of pixels
        img_height (int): Height of the input image in number of pixels

    Returns:
        int: Number of transitions allowed by the image dimension
    """

    limit_dim = min(img_width, img_height)
    n = 0

    while limit_dim % 2 == 0 and limit_dim > 2:
        limit_dim /= 2
        n += 1
    else:
        print("input dimension cannot be downsampled.")

    return n


def concatenate(inputs):
    """ Concatenation function that can handle input of length 1 properly

    Args:
        inputs (np.ndarray or list[np.ndarray]): Intermediate tensor or list of tensors

    Returns:
        np.ndarray: Tensor of properly concatenated input(s)
    """
    if isinstance(inputs, list):
        if len(inputs) > 1:
            outputs = Concatenate()(inputs)
        else:
            outputs = inputs[0]
        return outputs
    else:
        return inputs


def dense_conv_2d(inputs, growth_rate=48, bottleneck_factor=1):
    """ Convolution layer with dense connections.

    Args:
        inputs (np.ndarray or list[np.ndarray]): Intermediate tensor or list of tensors
        growth_rate (int): Multiplier for the number of filters in intermediate blocks
        bottleneck_factor (int): Ratio of number of filters in bottleneck block vs. previous block

    Returns:
        list[np.ndarray, np.ndarray]: List of two elements with first element being the output tensor of bottleneck
        block representing the dense connection, and the second being output tensor from a normal Conv2D layer.
    """
    concat = concatenate(inputs)
    bottle_neck = Conv2D(bottleneck_factor*growth_rate, (1, 1),
                         activation="selu", padding='same', kernel_initializer="lecun_normal")(concat)
    conv = Conv2D(growth_rate, (3, 3),
                  activation="selu", padding='same', kernel_initializer="lecun_normal")(bottle_neck)

    outputs = [bottle_neck, conv]

    return outputs


def dense_conv_block(inputs, growth_rate=48, n_layers=1, bottleneck_factor=1):
    """ Convolution block dense skip connections

    Args:
        inputs (np.ndarray or list[np.ndarray]): Intermediate tensor or list of tensors
        growth_rate (int): Multiplier for the number of filters in intermediate blocks
        n_layers (int): Supposed to be the number of convolution layers in this dense convolution block, but here it is
        just the multiplier of filter number growth.
        bottleneck_factor (int): Ratio of number of filters in bottleneck block vs. previous block

    Returns:
        list[np.ndarray, np.ndarray]: output of the dense convolution block.

    """
    n_layers = min(n_layers, 3)
    n_layers = max(n_layers, 1)
    outputs = dense_conv_2d(inputs, growth_rate * n_layers, bottleneck_factor)
    return outputs


def compression(inputs, compression_factor=0.5):
    """ Using 1x1 Conv2D filters to reduce number of model parameter

    Args:
        inputs (np.ndarray or list[np.ndarray]): Intermediate tensor or list of tensors
        compression_factor (float): Factor for shrinking number of filters from previous block

    Returns:
        np.ndarray: Compressed tensor from the 1x1 Conv2D layer
    """
    concat = concatenate(inputs)
    n_channels = int(concat.shape[-1])
    compression_filters = int(n_channels * compression_factor)
    outputs = Conv2D(compression_filters, (1, 1),
                     activation="selu", padding='same', kernel_initializer="lecun_normal")(concat)

    return outputs


def transition_down(inputs, compression_factor=0.5, pool_size=2):
    """ Downsampling module with a max pooling layer

    Args:
        inputs (np.ndarray or list[np.ndarray]): Intermediate tensor or list of tensors
        compression_factor (float): Factor for shrinking number of filters from previous block
        pool_size (int): Size of the pooling filter (window scanning through intermediate tensors)

    Returns:
        list[np.ndarray]: List of one tensor that was pooled and then compressed
    """
    concat = concatenate(inputs)
    pooled = MaxPooling2D(pool_size)(concat)

    n_channels = int(concat.shape[-1])
    compression_filters = int(n_channels * compression_factor)
    compression_1x1 = Conv2D(compression_filters, (1, 1),
                             activation="selu", padding='same', kernel_initializer="lecun_normal")(pooled)
    outputs = [compression_1x1]

    return outputs


def transition_up(inputs, compression_factor=0.5):
    """ Upsampling module with a transpose convolution layer

    Args:
        inputs (np.ndarray or list[np.ndarray]): Intermediate tensor or list of tensors
        compression_factor (float): Factor for shrinking number of filters from previous block

    Returns:
        list[np.ndarray]: List of one tensor that was compressed and then upsampled
    """
    concat = concatenate(inputs)

    n_channels = int(concat.shape[-1])
    compression_filters = int(n_channels * compression_factor)
    # ensure that compression filters are a multiple of 4
    possible_values = np.arange(0, 10000, 4)
    idx = np.argmin(np.abs(compression_filters - possible_values))
    compression_filters = possible_values[idx]

    compression_1x1 = Conv2D(compression_filters, (1, 1),
                             activation="selu", padding='same', kernel_initializer="lecun_normal")(concat)
    upsampled = Conv2DTranspose(compression_filters, (3, 3), strides=(2, 2),
                                activation="selu", padding='same', kernel_initializer="lecun_normal")(compression_1x1)
    outputs = [upsampled]

    return outputs


def frontend(inputs, growth_rate=48, n_downsample=1, compression_factor=0.5, bottleneck_factor=1):
    """ Initial block in StackedDenseNet for preliminary transformations before entering DenseNet modules.

    Args:
        inputs (np.ndarray or list[np.ndarray]): Intermediate tensor or list of tensors
        growth_rate (int): Multiplier for the number of filters in intermediate blocks
        n_downsample (int): Number of transition down blocks applied
        compression_factor (float): Factor for shrinking number of filters from previous block
        bottleneck_factor (int): Ratio of number of filters in bottleneck block vs. previous block

    Returns:
        list[np.ndarray]: List of one concatenated tensor

    """
    conv_7x7 = Conv2D(growth_rate, (7, 7), strides=(2, 2),
                      activation="selu", padding='same', kernel_initializer="lecun_normal")(inputs)
    pooled_inputs = MaxPooling2D(pool_size=2)(inputs)

    outputs = [pooled_inputs, conv_7x7]
    residual_outputs = []

    for idx in range(n_downsample - 1):
        outputs = dense_conv_block(outputs, growth_rate, (idx+1), bottleneck_factor)
        concat_outputs = concatenate(outputs)
        outputs = [concat_outputs]

        # Pool each dense layer to match output size
        pooled_outputs = transition_down(outputs, compression_factor, pool_size=2 ** (n_downsample - 1 - idx))
        residual_outputs.append(concatenate(pooled_outputs))

        outputs = transition_down(outputs, compression_factor)

    outputs = dense_conv_block(outputs, growth_rate, (n_downsample - 1), bottleneck_factor)
    outputs = concatenate(outputs)
    residual_outputs.append(outputs)
    residual_outputs = [
        compression(res, compression_factor) for res in residual_outputs
    ]
    outputs = concatenate(residual_outputs)
    return [outputs]


def densenet(inputs, growth_rate=64, n_downsample=1, downsample_factor=0, compression_factor=0.5, bottleneck_factor=1):
    """ Single DenseNet module with an Encoder and an Decoder segment.

    Args:
        inputs (np.ndarray or list[np.ndarray]): Intermediate tensor or list of tensors
        growth_rate (int): Multiplier for the number of filters in intermediate blocks
        n_downsample (int): Number of transition down blocks applied
        downsample_factor (int): Downsample factor for calculation of n_layer in dense_conv_block
        compression_factor (float): Factor for shrinking number of filters from previous block
        bottleneck_factor (int): Ratio of number of filters in bottleneck block vs. previous block

    Returns:
        list[np.ndarray]: List of one concatenated tensor

    """
    n_upsample = n_downsample

    residual_outputs = [concatenate(inputs)]
    outputs = transition_down(inputs, compression_factor)

    # Encoder
    for idx in range(n_downsample - 1):
        outputs = dense_conv_block(outputs, growth_rate, (idx+1+downsample_factor), bottleneck_factor)
        concat_outputs = concatenate(outputs)
        outputs = [concat_outputs]
        residual_outputs.append(concat_outputs)
        outputs = transition_down(outputs, compression_factor)

    residual_outputs.append(concatenate(outputs))
    outputs = dense_conv_block(outputs, growth_rate, downsample_factor, bottleneck_factor)

    # Compress the feature maps for residual connections
    residual_outputs = residual_outputs[::-1]
    residual_outputs = [
        compression(res, compression_factor) for res in residual_outputs
    ]

    # Decoder
    for idx in range(n_upsample):
        outputs.append(residual_outputs[idx])
        n_layers = n_upsample - (idx+1+downsample_factor) + 1
        outputs = dense_conv_block(outputs, growth_rate, n_layers, bottleneck_factor)
        outputs = transition_up(outputs, compression_factor)
    outputs.append(residual_outputs[-1])
    outputs = dense_conv_block(outputs, growth_rate, downsample_factor, bottleneck_factor)

    return [concatenate(outputs)]


def output_channels(inputs, num_joints):
    """ Output block applied to final and intermediate outputs

    Args:
        inputs (np.ndarray or list[np.ndarray]): Intermediate tensor or list of tensors
        num_joints (int): Number of joints of the dataset animal

    Returns:
        np.ndarray: Final or intermediate model output tensor
        np.ndarray: Output tensor for downstream blocks

    """
    concat = concatenate(inputs)
    n_channels = int(concat.shape[-1])
    compression_filters = int(n_channels * 0.5)
    # ensure that compression filters are a multiple of 4
    possible_values = np.arange(0, 10000, 4)
    idx = np.argmin(np.abs(compression_filters - possible_values))
    compression_filters = possible_values[idx]

    compression_1x1 = Conv2D(compression_filters, (1, 1),
                             activation="selu", padding='same', kernel_initializer="lecun_normal")(concat)
    #
    upsampled = Conv2DTranspose(compression_filters, (3, 3), strides=(2, 2),
                                activation="selu", padding='same', kernel_initializer="lecun_normal")(compression_1x1)
    model_outputs = Conv2D(num_joints, (1, 1), padding="same", activation="linear")(upsampled)
    outputs = Conv2D(num_joints, (1, 1), padding="same", activation="linear")(concat)
    return model_outputs, outputs


# Main Model -------------------------------------------------------------------------------------------------------- #

def stacked_densenet(img_width, img_height, num_joints, learning_rate, num_stage,
                     growth_rate=48, bottleneck_factor=1, compression_factor=0.5):
    """ StackedDenseNet Model fully assembled

    Args:
        img_width (int): Width of the input image in number of pixels
        img_height (int): Height of the input image in number of pixels
        num_joints (int): Number of joints of the dataset animal
        learning_rate (float): Scalar value indicating speed of gradient update
        num_stage (int): Number of stacks of DenseNet module
        growth_rate (int): Multiplier for the number of filters in intermediate blocks
        bottleneck_factor (int): Ratio of number of filters in bottleneck block vs. previous block
        compression_factor (float): Factor for shrinking number of filters from previous block

    Returns:
        Keras model: untrained StackedDenseNet model

    """
    n_transitions = num_transition(img_width, img_height)
    input_img = Input(shape=(img_height, img_width, 3))
    front_outputs = frontend(inputs=input_img,
                             growth_rate=growth_rate,
                             n_downsample=0,
                             compression_factor=compression_factor,
                             bottleneck_factor=bottleneck_factor)

    n_downsample = n_transitions - 2
    outputs = front_outputs
    model_outputs, out_channels = output_channels(outputs, num_joints)
    model_outputs_list = [model_outputs]
    outputs.append(BatchNormalization()(out_channels))

    for idx in range(num_stage):
        outputs = densenet(outputs, growth_rate, n_downsample, 0, compression_factor, bottleneck_factor)
        outputs.append(concatenate(front_outputs))
        outputs.append(BatchNormalization()(out_channels))
        model_outputs, out_channels = output_channels(outputs, num_joints)
        model_outputs_list.append(model_outputs)

    model = Model(inputs=input_img, outputs=model_outputs_list)
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
