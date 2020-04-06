import os
import cv2
import numpy as np
from data.lbl_proc import hml_read
from data.lbl_proc import hm_optical_warp

"""Function list:
# info extraction functions:
    get_filenames(folder):  Get a list of sorted filename (without folder path and extension) from the given folder path
    path_to_img(img_path):  Read the image specified by the given image path
    path_to_hm(hm_path, max_val):  Read the heatmap pickle file specified by the given heatmap path
    images_to_tensor(img_folder):  Collect all images in the given folder into a {4D} tensor and keep a list of 
    sorted filenames
    hm_tensor_to_lst(tensor):  Convert the given heatmap tensor to a list of values representing x or y coordinate of 
    maximum value on each slice of the heatmap tensor.
    get_range_tensor(range_tensor_folder):  Collect all optical flow preprocessed data in the given folder into a {5D} 
    tensor and keep a list of sorted filenames
    
# optical flow related data manipulation functions:
    path_to_image_range_tensor(name, img_folder, frame_range, skip_ratio):  Get an tensor of images from adjacent frames
    multiple_optical_warp(img_tensor, hm_tensor, frame_range):  Compute optical flow warps for a heatmap tensor
    valid_frame(name, names, frame_range, skip_ratio):  Check if the frame, specified by name, has adjacent frames
    img_tensor_to_warped_hm_tensor(dataset_img_tensor, base_model, frame_range, skip_ratio, names):  Converts all images
    of the given dataset tensor into warped optical flow tensors using the provided base model
"""


# info extraction functions ----------------------------------------------------------------------------------------- #

def get_filenames(folder):
    """ Get a list of sorted filenames (without folder path and extension) from the given folder path

    Args:
        folder (str): Complete folder path

    Returns:
        list[str]: Sorted list of filenames (without folder path and extension)
    """
    files = os.listdir(folder)
    names = []

    for file_name in files:
        name = os.path.splitext(file_name)[0]  # Get name without PATH and EXT
        names.append(name)
    names.sort()
    return names


def path_to_img(img_path):
    """  Read the image specified by the given image path

    Args:
        img_path (str): Complete image path

    Returns:
        np.ndarray: {2D, 3D} Image tensor
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    if img is not None:
        img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    else:
        print("Problem Reading %s" % img_path)

    return img


def path_to_hm(hm_path, max_val):
    """ Read the heatmap pickle file specified by the given heatmap path

    Args:
        hm_path (str): Complete heatmap path
        max_val (float): Maximum value for the extracted heatmaps

    Returns:
        np.ndarray: {3D} Heatmap tensor
    """
    heatmap_dict_arr = hml_read(hm_path)
    heatmap_arr = []

    for table in heatmap_dict_arr:
        hm_slice = table['heatmap']
        slice_max = np.max(hm_slice, axis=None)

        if slice_max > 0:
            multiplier = max_val/slice_max
            hm_slice = np.multiply(hm_slice, multiplier)
        
        heatmap_arr.append(hm_slice)

    heatmap_tensor = np.dstack(heatmap_arr)

    return heatmap_tensor


def images_to_tensor(img_folder):
    """ Collect all images in the given folder into a {4D} tensor and keep a list of sorted filenames

    Args:
        img_folder (str): Complete image folder path

    Returns:
        np.ndarray: {4D} Tensor of all images in the given image folder
        list[str]: Sorted list of image filenames, without folder path and extension
    """

    names = get_filenames(img_folder)

    images = []
    for name in names:
        img_path = os.path.join(img_folder, name) + ".png"
        img = path_to_img(img_path)
        images.append(img)

    tensor = np.array(images)

    return tensor, names


def hm_tensor_to_lst(tensor):
    """ Convert the given heatmap tensor to a list of values representing x or y coordinate of maximum value
    on each slice of the heatmap tensor.

    Args:
        tensor (np.ndarray): {3D} Heatmap tensor

    Returns:
        list[int]: Even index of list represent x coordinates of maximum values and odd index represent y
        coordinates of maximum values
    """
    num_joint = tensor.shape[-1]
    lst = []

    for i in range(num_joint):
        hm_slice = tensor[..., i]
        y, x = np.unravel_index(np.argmax(hm_slice), hm_slice.shape)
        lst.append(int(x))
        lst.append(int(y))

    return lst


def get_range_tensor(range_tensor_folder):
    """ Collect all optical flow preprocessed data in the given folder into a {5D} tensor and
    keep a list of sorted filenames

    Args:
        range_tensor_folder (str): Complete optical flow preprocessed dataset folder path

    Returns:
        np.ndarray: {5D} tensor of all optical flow preprocessed data of a given dataset
        list[str]: Sorted list of filenames, without folder path and extension
    """
    names = get_filenames(range_tensor_folder)
    tensors = []

    for name in names:
        tensor_path = os.path.join(range_tensor_folder, name) + ".pkl"
        tensor = hml_read(tensor_path)
        tensors.append(tensor)

    range_tensor = np.array(tensors)

    return range_tensor, names


# optical flow related data manipulation functions ------------------------------------------------------------------- #

def path_to_image_range_tensor(name, img_folder, frame_range, skip_ratio):
    """ Get an tensor of images from adjacent frames

    Args:
        name (str): Filename, without extension or folder path, of the reference frame
        img_folder (str): Complete image folder path
        frame_range (int): Number of adjacent frames to consider from before and after the reference frame
        skip_ratio (int): Sampling ratio for extracting adjacent frames. E.g. 1 means extract every frame, 2 means
        extract every two frames, etc.

    Returns:
        np.ndarray: {4D} tensor containing all adjacent frames (including the reference frame itself)
    """
    code_lst = name.split("_")
    prefix_lst = code_lst[:-1]
    prefix = "_".join(prefix_lst)

    ref_frame_num = int(code_lst[-1])

    frame_nums = []
    # load frame numbers in order
    for i in range(frame_range, 0, -1):
        frame_nums.append(ref_frame_num - i*skip_ratio)

    frame_nums.append(ref_frame_num)

    for i in range(1, frame_range+1):
        frame_nums.append(ref_frame_num + i*skip_ratio)

    tensor_lst = []

    for num in frame_nums:
        frame_name = prefix + "_" + format(num, "05d")
        img_path = os.path.join(img_folder, frame_name) + ".png"
        img = path_to_img(img_path)  # (h,w,3) tensor

        # valid frame only check for the range, not the individual frames
        if img is None:
            # if one frame is missing, then we replace with previous frame, this happens because some frame is taken
            # out for missing labels.
            frame_name = prefix + "_" + format(num-1, "05d")
            img_path = os.path.join(img_folder, frame_name) + ".png"
            img = path_to_img(img_path)  # (h,w,3) tensor

        tensor_lst.append(img)

    if len(tensor_lst) != len(frame_nums):
        print("Some frames missing within the frame range!")

    return np.array(tensor_lst)


def multiple_optical_warp(img_range_tensor, hm_tensor, frame_range):
    """ Compute optical flow warps for a heatmap tensor

    Args:
        img_range_tensor (np.ndarray): {4D} Tensor containing all adjacent frames within frame range
        hm_tensor (np.ndarray): {3D} Heatmap tensor of reference frame
        frame_range (int): Number of adjacent frames to consider from before and after the reference frame

    Returns:
        np.ndarray: {3D} Heatmap tensor warped by optical flow
    """
    num_frame = 2*frame_range + 1
    ref_img = img_range_tensor[frame_range]

    for i in range(num_frame):
        other_img = img_range_tensor[i]
        hm_tensor[i] = hm_optical_warp(other_img, ref_img, hm_tensor[i])  # warp each slice of hm_tensor

    return hm_tensor


def valid_frame(name, names, frame_range, skip_ratio):
    """ Check if the frame, specified by name, has adjacent frames

    Args:
        name (str): Filename, without extension or folder path, of the reference frame
        names (list[str]): Sorted list of filename, without folder path or extension
        frame_range (int): Number of adjacent frames to consider from before and after the reference frame
        skip_ratio (int): Sampling ratio for extracting adjacent frames. E.g. 1 means extract every frame, 2 means
        extract every two frames, etc.

    Returns:
        bool: indicate whether the frame is valid or not
    """
    code_lst = name.split("_")
    prefix_lst = code_lst[:-1]
    prefix = "_".join(prefix_lst)

    frame_num = int(code_lst[-1])
    first_frame = frame_num - frame_range*skip_ratio
    last_frame = frame_num + frame_range*skip_ratio

    if first_frame < 0:
        return False

    first_frame_name = prefix + "_" + format(first_frame, "05d")
    last_frame_name = prefix + "_" + format(last_frame, "05d")

    if {first_frame_name, last_frame_name}.issubset(set(names)):
        return True

    return False


def img_tensor_to_warped_hm_tensor(dataset_img_tensor, base_model, frame_range, skip_ratio, names):
    """ Converts all images of the given dataset tensor into warped optical flow tensors using the provided base model

    Args:
        dataset_img_tensor (np.ndarray): {4D} Tensor containing all images of the given dataset
        base_model (Keras model): a trained base model to make initial predictions on image tensors.
        frame_range (int): Number of adjacent frames to consider from before and after the reference frame
        skip_ratio (int): Sampling ratio for extracting adjacent frames. E.g. 1 means extract every frame, 2 means
        extract every two frames, etc.
        names (list[str]): Sorted list of filename for the given dataset, without folder path or extension

    Returns:
        np.ndarray: {4D} Tensor containing optical flow warped heatmap of all images of the given dataset
        list[str]: Sorted list of filename of the warped heatmaps, without folder path or extension
    """
    warped_hm_names = []
    warped_hm_lst = []

    init_idx = frame_range*skip_ratio
    end_idx = len(dataset_img_tensor)-frame_range*skip_ratio

    for i in range(init_idx, end_idx):
        first_frame_idx = i-frame_range*skip_ratio
        last_frame_idx = i+frame_range*skip_ratio+1
        cur_img_tensor = dataset_img_tensor[first_frame_idx:last_frame_idx:skip_ratio]

        stages = base_model.predict(cur_img_tensor)
        hm_tensor = stages[-1]

        warped_hm_tensor = multiple_optical_warp(cur_img_tensor, hm_tensor, frame_range)
        warped_hm_lst.append(warped_hm_tensor)
        warped_hm_names.append(names[i])

    output_tensor = np.array(warped_hm_lst)

    return output_tensor, warped_hm_names
