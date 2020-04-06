import os
import numpy as np
from data.lbl_proc import hml_read
from data.dataset_func import get_filenames, path_to_hm, path_to_img

"""Function list:
# data generators:
    hm_datagen(img_folder, lbl_folder, batch_size, max_val, num_stage):  Data generator for heatmap models  
    fast_flow_datagen(tensor_folder, lbl_folder, batch_size, max_val):  Data generator for optical flow models       
    
# model training functions:
    train_generator(train_img_folder, train_lbl_folder, ver_img_folder, ver_lbl_folder,model, num_stage, batch_size, 
    epochs, max_val):  Training function for heatmap models
    train_fast_flow_generator(train_tensor_folder, train_lbl_folder, ver_tensor_folder, ver_lbl_folder, flow_model, 
    batch_size, epochs, max_val):  Training function for optical flow models
"""


# data generators --------------------------------------------------------------------------------------------------- #

def hm_datagen(img_folder, lbl_folder, batch_size, max_val, num_stage):
    """ Data generator for heatmap models

    Args:
        img_folder (str): Complete image folder path
        lbl_folder (str): Complete heatmap label folder path
        batch_size (int): Number of image-label pairs to be included in each batch
        max_val (float): Maximum value for the label heatmaps
        num_stage (int): Number of stages of the training model

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple of {4D} tensor of images in the batch and {4D} tensor of heatmap labels
        for those images, respectively
    """
    names = get_filenames(img_folder)
    while True:
        batch_names = np.random.choice(a=names, size=batch_size)

        batch_img = []
        batch_lbl = []

        for name in batch_names:
            img_path = os.path.join(img_folder, name) + ".png"
            hm_path = os.path.join(lbl_folder, name) + ".pkl"

            if os.path.isfile(hm_path):
                img = path_to_img(img_path)  # (h,w,3) tensor
                lbl = path_to_hm(hm_path, max_val)  # (h,w, num_joints) tensor

                batch_img.append(img)
                batch_lbl.append(lbl)
            else:

                print('Missing label file for [%s], this datum will be EXCLUDED form training!' % name)

        batch_X = np.array(batch_img)
        batch_lbl = np.array(batch_lbl)

        batch_y = [batch_lbl] * num_stage  # for single stage the output will be [batch_lbl]

        yield (batch_X, batch_y)


def fast_flow_datagen(tensor_folder, lbl_folder, batch_size, max_val):
    """ Data generator for optical flow models

    Args:
        tensor_folder (str): Complete warped tensor folder path
        lbl_folder (str): Complete heatmap label folder path
        batch_size (int): Number of image-label pairs to be included in each batch
        max_val (float): Maximum value for the label heatmaps

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple of {4D} tensor of optical flow warped heatmap in the batch and
        {4D} tensor of heatmap labels for those images, respectively
    """
    names = get_filenames(tensor_folder)

    while True:
        batch_names = np.random.choice(a=names, size=batch_size)

        batch_range_tensor = []
        batch_hm = []

        for name in batch_names:
            tensor_path = os.path.join(tensor_folder, name) + ".pkl"
            range_tensor = hml_read(tensor_path)

            hm_path = os.path.join(lbl_folder, name) + ".pkl"

            if os.path.isfile(hm_path):
                hm = path_to_hm(hm_path, max_val)  # (h,w,num_joints) tensor
                batch_range_tensor.append(range_tensor)
                batch_hm.append([hm])
            else:
                print('Missing label file for [%s], this datum will be EXCLUDED form training!' % name)

        batch_X = np.array(batch_range_tensor)
        batch_y = np.array(batch_hm)

        yield (batch_X, batch_y)


# model training functions ------------------------------------------------------------------------------------------ #

def train_generator(train_img_folder, train_lbl_folder, ver_img_folder, ver_lbl_folder,
                    model, num_stage, batch_size, epochs, max_val):
    """ Training function for heatmap models

    Args:
        train_img_folder (str): Complete train set image folder path
        train_lbl_folder (str): Complete train set heatmap label folder path
        ver_img_folder (str): Complete verification set image folder path
        ver_lbl_folder (str): Complete verification set heatmap label folder path
        model (Keras model): Untrained model
        num_stage (int): Number of stages of the untrained model
        batch_size (int): Number of image-label pairs to be included in each training batch
        epochs (int): Number of time the entire dataset will be passed through during training
        max_val (float): Maximum value for the label heatmaps

    Returns:
        Keras model: trained model
        Keras model history: history file that contain all of the measured metric and intermediate value throughout
        the training process
    """
    train_gen = hm_datagen(train_img_folder, train_lbl_folder, batch_size, max_val, num_stage)
    print("training images generator ready")

    ver_gen = hm_datagen(ver_img_folder, ver_lbl_folder, batch_size, max_val, num_stage)
    print("verification images generator ready")

    num_train_data = len(get_filenames(train_img_folder))
    num_ver_data = len(get_filenames(ver_img_folder))

    train_epoch_step = num_train_data // batch_size
    ver_epoch_step = num_ver_data // batch_size

    train_out = model.fit_generator(generator=train_gen,
                                    validation_data=ver_gen,
                                    epochs=epochs,
                                    steps_per_epoch=train_epoch_step,
                                    validation_steps=ver_epoch_step,
                                    max_queue_size=30, workers=20, use_multiprocessing=True)

    history = train_out.history

    return model, history


def train_fast_flow_generator(train_tensor_folder, train_lbl_folder, ver_tensor_folder, ver_lbl_folder,
                              flow_model, batch_size, epochs, max_val):
    """ Training function for heatmap models

    Args:
        train_tensor_folder (str): Complete train set warped tensor folder path
        train_lbl_folder (str): Complete train set heatmap label folder path
        ver_tensor_folder (str): Complete verification set warped tensor folder path
        ver_lbl_folder (str): Complete verification set heatmap label folder path
        flow_model (Keras model): Untrained optical flow model
        batch_size (int): Number of image-label pairs to be included in each training batch
        epochs (int): Number of time the entire dataset will be passed through during training
        max_val (float): Maximum value for the label heatmaps

    Returns:
        Keras model: trained optical flow model
        Keras model history: history file that contain all of the measured metric and intermediate value throughout
        the training process
    """
    train_gen = fast_flow_datagen(train_tensor_folder, train_lbl_folder, batch_size, max_val)
    print("training tensor generator ready")

    ver_gen = fast_flow_datagen(ver_tensor_folder, ver_lbl_folder, batch_size, max_val)
    print("verification tensor generator ready")

    num_train_data = len(get_filenames(train_tensor_folder))
    num_ver_data = len(get_filenames(ver_tensor_folder))

    train_epoch_step = num_train_data // batch_size
    ver_epoch_step = num_ver_data // batch_size

    train_out = flow_model.fit_generator(generator=train_gen,
                                         validation_data=ver_gen,
                                         epochs=epochs,
                                         steps_per_epoch=train_epoch_step,
                                         validation_steps=ver_epoch_step,
                                         max_queue_size=30, workers=20, use_multiprocessing=True)

    history = train_out.history

    return flow_model, history
