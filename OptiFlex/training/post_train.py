import os
import cv2
import json
import pickle
import collections
import numpy as np
from OptiFlex.utils.disp_func import img_hml_plt, img_jsl_plt
from OptiFlex.data.lbl_proc import hml_write
from OptiFlex.data.dataset_func import images_to_tensor, get_range_tensor


"""Function list:
# get animal information:
    get_color_list(dataset_animal): Get the pre-defined list of joint color dictionary for the given animal
    get_joint_list(dataset_animal): Get the pre-defined list of joint names list for the given animal

# data and model post processing:   
    create_json_template(joint_list, lbl_width, lbl_height): Generate a template for JSON results
    vec_to_json_arr(pred_vec, joint_list): Fill out JSON template with a given prediction results array 
    hm_tensor_to_lst(tensor, flow): Identify peak value location of each heatmap and store the values in a list
    tensor_to_heatmap_file(tensor, joint_list): Fill out pickle file template with a given prediction result tensor
    save_record(model, history, model_name, save_dir): Save model and history files to a given directory
    
# create result folders:
    create_heatmap_files(tensor_preds, names, dataset_animal, output_dir): Store heatmap predictions in a folder
    create_output_files(predictions, names, dataset_animal, output_dir): Store JSON predictions in a folder
    create_labeled_hm_img_folder(hm_folder, img_folder, dataset_animal,  output_dir): Store heatmap labelled images
    create_labeled_img_folder(lbl_folder, img_folder, dataset_animal, output_dir): Store JSON labelled images

# inference and PCK
    inference(model, num_stage, dataset_animal, test_img_folder, pred_js_folder, pred_hm_folder, dataset_folder):
    Use trained model to make predictions on test images
    fast_flow_inference(fast_flow_model, dataset_animal, test_tensor_folder, test_img_folder, pred_js_folder, 
    pred_hm_folder, dataset_folder): Use trained optical flow model to make predictions on pre-processed test tensors
    print_pck(eval_out):  Print PCK evaluation output
"""


# get animal information -------------------------------------------------------------------------------------------- #

def get_color_list(dataset_animal):
    """ Get the pre-defined list of joint color dictionary for the given animal

    Args:
        dataset_animal (str): One of "mouse", "fruitfly", "zebrafish", "monkey"

    Returns:
         dict[str, tuple[int, int, int]]: Dictionary with animal joint name as key and tuple of RGB color as values
    """
    if dataset_animal == "mouse":
        mouse_color = {
                        "Front Right Paw": (255, 0, 0), "Hind Right Paw": (255, 255, 0),
                        "Front Left Paw": (0, 255, 0), "Hind Left Paw": (0, 255, 255),
                        "Snout": (0, 0, 255),
                        "Tail 01": (255, 0, 255), "Tail 02": (255, 0, 255), "Tail 03": (255, 0, 255)
                      }
        return mouse_color

    elif dataset_animal == "fruitfly":
        fruitfly_color = {
                           "head": (76, 109, 81), "eyeL": (240, 214, 234), "eyeR": (240, 214, 234),
                           "neck": (255, 0, 5), "thorax": (238, 238, 130), "abdomen": (0, 0, 100),
                           "forelegR1": (248, 121, 131), "forelegR2": (238, 77, 32), "forelegR3": (237, 57, 41),
                           "forelegR4": (139, 0, 0),
                           "midlegR1": (255, 205, 250), "midlegR2": (252, 131, 232), "midlegR3": (255, 0, 211),
                           "midlegR4": (218, 32, 165),
                           "hindlegR1": (161, 241, 202), "hindlegR2": (31, 254, 117), "hindlegR3": (16, 166, 52),
                           "hindlegR4": (31, 94, 48),
                           "forelegL1": (79, 176, 255), "forelegL2": (135, 107, 169), "forelegL3": (0, 78, 106),
                           "forelegL4": (27, 62, 77),
                           "midlegL1": (178, 190, 132), "midlegL2": (133, 136, 96), "midlegL3": (148, 211, 0),
                           "midlegL4": (86, 92, 60),
                           "hindlegL1": (255, 102, 153), "hindlegL2": (254, 94, 111), "hindlegL3": (196, 16, 98),
                           "hindlegL4": (150, 0, 75),
                           "wingL": (255, 240, 255), "wingR": (255, 240, 255)
                         }
        return fruitfly_color

    elif dataset_animal == "monkey":
        monkey_color = {
                         "upperlip1": (255, 0, 0), "upperlip2": (255, 255, 0), "lowerlip1": (0, 255, 0),
                         "lowerlip2": (0, 255, 255),
                         "brow": (0, 0, 255), "lickspout": (255, 255, 255), "tongue": (255, 0, 255)
                       }
        return monkey_color

    elif dataset_animal == "zebrafish":
        zebrafish_color = {
                            "zf_01": (255, 0, 0), "zf_02": (255, 255, 0), "zf_03": (0, 255, 0),
                            "zf_04": (0, 255, 255), "zf_05": (0, 0, 255), "zf_06": (255, 0, 255),
                            "zf_07": (255, 145, 145), "zf_08": (255, 210, 105), "zf_09": (53, 128, 10),
                            "zf_10": (84, 160, 227), "zf_11": (189, 9, 184), "zf_12": (87, 145, 55)
                          }
        return zebrafish_color

    else:
        print("Dataset for " + dataset_animal + " is not currently available.")
        print("Please choose from mouse, monkey, fruitfly, or zebrafish")


def get_joint_list(dataset_animal):
    """ Get the pre-defined list of joint names list for the given animal

    Args:
        dataset_animal (str): One of "mouse", "fruitfly", "zebrafish", "monkey"

    Returns:
         list[str]: List of joint names for the given animal
    """
    mouse = [
        "Front Right Paw", "Hind Right Paw", "Front Left Paw", "Hind Left Paw", "Snout",
        "Tail 01", "Tail 02", "Tail 03"
    ]

    fruitfly = [
        "head", "eyeL", "eyeR", "neck", "thorax", "abdomen",
        "forelegR1", "forelegR2", "forelegR3", "forelegR4", "midlegR1", "midlegR2", "midlegR3", "midlegR4",
        "hindlegR1", "hindlegR2", "hindlegR3", "hindlegR4", "forelegL1", "forelegL2", "forelegL3", "forelegL4",
        "midlegL1", "midlegL2", "midlegL3", "midlegL4", "hindlegL1", "hindlegL2", "hindlegL3", "hindlegL4",
        "wingL", "wingR"
    ]

    zebrafish = [
        "zf_01", "zf_02", "zf_03", "zf_04", "zf_05", "zf_06", "zf_07", "zf_08", "zf_09", "zf_10", "zf_11", "zf_12"
    ]

    monkey = [
        "upperlip1", "upperlip2", "lowerlip1", "lowerlip2", "brow", "lickspout", "tongue"
    ]

    if dataset_animal == "mouse":
        return mouse
    elif dataset_animal == "fruitfly":
        return fruitfly
    elif dataset_animal == "zebrafish":
        return zebrafish
    elif dataset_animal == "monkey":
        return monkey
    else:
        print("Dataset for " + dataset_animal + " is not currently available.")
        print("Please choose from mouse, monkey, fruitfly, or zebrafish")


#  data and model post processing ----------------------------------------------------------------------------------- #

def create_json_template(joint_list, lbl_width=1, lbl_height=1):
    """ Generate a template for JSON results

    Args:
        joint_list (list[str]): List of joint names
        lbl_width (int): Width of label bounding box; 1 if label is a point
        lbl_height (int): Height of label bounding box; 1 if label is a point

    Returns:
        list[dict]: List of dictionary that will be the template of final JSON label file.
    """
    num_joints = len(joint_list)
    joint_template = {"left": None, "top": None, "width": lbl_width, "height": lbl_height, "label": None}
    labels_template = [None] * num_joints

    for i in range(num_joints):
        joint = joint_template.copy()
        joint["label"] = joint_list[i]
        labels_template[i] = joint

    return labels_template


def vec_to_json_arr(pred_vec, joint_list):
    """ Fill out JSON template with a given prediction results array

    Args:
        pred_vec (list(float)): List of floats that represent a vector of joint location predictions
        joint_list (list[str]): List of joint names

    Returns:
        list[dict]: List of dictionary that is the JSON template filled out with prediction results.
    """
    json_arr = create_json_template(joint_list)
    num_joints = len(joint_list)

    for i in range(num_joints):
        json_arr[i]["left"] = pred_vec[2 * i]
        json_arr[i]["top"] = pred_vec[(2 * i) + 1]

    return json_arr


def hm_tensor_to_lst(tensor, flow=False):
    """ Identify peak value location of each heatmap and store the values in a list

    Args:
        tensor (np.ndarray): {3D} Heatmap tensor
        flow (bool): Indict whether or not this is used for optical flow result conversion

    Returns:
        list[int]: Even index of list represent x coordinates of maximum values and odd index represent y
        coordinates of maximum values
    """
    lst = []
    joint_num = tensor.shape[-1]

    if flow:
        tensor = tensor[0]

    for i in range(joint_num):
        hm_slice = tensor[..., i]
        y, x = np.unravel_index(np.argmax(hm_slice), hm_slice.shape)
        lst.append(int(x))
        lst.append(int(y))

    return lst


def tensor_to_heatmap_file(tensor, joint_list):
    """ Fill out pickle file template with a given prediction result tensor

    Args:
        tensor (np.ndarray): {3D} Heatmap tensor
        joint_list (list[str]): List of joint names

    Returns: list[dict]: List of dictionary that is the pickle file template filled out with prediction results.

    """
    lst = []
    joint_num = tensor.shape[-1]

    for i in range(joint_num):
        table = {"label": joint_list[i], "heatmap": tensor[..., i]}
        lst.append(table)

    return lst


def save_record(model, history, model_name="model", save_dir="../../model"):
    """ Save model and history files to a given directory

    Args:
        model (Keras model): model object of a given trained model
        history (Keras model history): model history object of a given trained model
        model_name (str): Name of the model file
        save_dir (str): Complete saving directory path
    """
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)

    print('Saved trained model at %s ' % model_path)

    history_path = save_dir + "/" + model_name + "History"
    with open(history_path, 'wb') as file_pi:
        pickle.dump(history, file_pi)

    print('Saved training history at %s ' % history_path)


# get animal information -------------------------------------------------------------------------------------------- #

def create_heatmap_files(tensor_preds, names, dataset_animal, output_dir="."):
    """ Store heatmap predictions in a folder

    Args:
        tensor_preds (np.ndarray): {4D} tensor with each slice of {3D} tensor being the heatmap prediction of an image
        names (list[str]): Sorted list of filenames for the prediction results
        dataset_animal (str): One of "mouse", "fruitfly", "zebrafish", "monkey"
        output_dir (str): Complete output directory path
    """
    path = os.path.join(output_dir, "pred_hm_lbl")
    joint_list = get_joint_list(dataset_animal)

    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)

    for i in range(len(tensor_preds)):
        hm_file = tensor_to_heatmap_file(tensor_preds[i], joint_list)
        hml_write(os.path.join(path, names[i] + '.pkl'), hm_file)


def create_output_files(predictions, names, dataset_animal, output_dir="."):
    """ Store JSON predictions in a folder

    Args:
        predictions (list[list]): list of prediction result array
        names (list[str]): Sorted list of filenames for the prediction results
        dataset_animal (str): One of "mouse", "fruitfly", "zebrafish", "monkey"
        output_dir (str): Complete output directory path
    """
    path = os.path.join(output_dir, "pred_js_lbl")
    joint_list = get_joint_list(dataset_animal)

    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)

    for i in range(len(predictions)):
        arr = vec_to_json_arr(predictions[i], joint_list)
        with open(os.path.join(path, names[i] + '.json'), 'w') as outfile:
            json.dump(arr, outfile)


def create_labeled_hm_img_folder(hm_folder, img_folder, dataset_animal,  output_dir="."):
    """ Store heatmap labelled images

    Args:
        hm_folder (str): Complete heatmap prediction results folder path
        img_folder (str): Complete test image folder path
        dataset_animal (str): One of "mouse", "fruitfly", "zebrafish", "monkey"
        output_dir (str): Complete output directory path
    """
    img_names = [img_name for img_name in os.listdir(img_folder) if img_name.endswith(".png")]
    hm_names = [hm_name for hm_name in os.listdir(hm_folder) if hm_name.endswith(".pkl")]
    img_names = sorted(img_names)
    hm_names = sorted(hm_names)
    path = os.path.join(output_dir, "hm_lbl_img")
    color_list = get_color_list(dataset_animal)

    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)

    if len(img_names) != len(hm_names):
        print("Number of Heatmaps and Images Do Not Match!")
        return None

    for i in range(len(img_names)):
        labeled_img, title = img_hml_plt(os.path.join(img_folder, img_names[i]),
                                         os.path.join(hm_folder, hm_names[i]), color_list=color_list)
        cv2.imwrite(os.path.join(path, title + ".jpg"), labeled_img)


def create_labeled_img_folder(lbl_folder, img_folder, dataset_animal, output_dir="."):
    """ Store JSON labelled images

    Args:
        lbl_folder (str): Complete JSON prediction results folder path
        img_folder (str): Complete test image folder path
        dataset_animal (str): One of "mouse", "fruitfly", "zebrafish", "monkey"
        output_dir (str): Complete output directory path
    """
    img_names = [img_name for img_name in os.listdir(img_folder) if img_name.endswith(".png")]
    lbl_names = [lbl_name for lbl_name in os.listdir(lbl_folder) if lbl_name.endswith(".json")]
    img_names = sorted(img_names)
    lbl_names = sorted(lbl_names)
    path = os.path.join(output_dir, "js_lbl_img")
    color_list = get_color_list(dataset_animal)

    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)

    if len(img_names) != len(lbl_names):
        print("Number of Labels and Images Do Not Match!")
        return None

    for i in range(len(img_names)):
        labeled_img, title = img_jsl_plt(os.path.join(img_folder, img_names[i]),
                                         os.path.join(lbl_folder, lbl_names[i]), color_list=color_list)
        cv2.imwrite(os.path.join(path, title + ".jpg"), labeled_img)


# get animal information -------------------------------------------------------------------------------------------- #

def inference(model, num_stage, dataset_animal, test_img_folder, pred_js_folder, pred_hm_folder, dataset_folder):
    """ Use trained model to make predictions on test images

    Args:
        model (Keras model): Trained model for inference predictions
        num_stage (int): Number of stages of the trained model
        dataset_animal (str): One of "mouse", "fruitfly", "zebrafish", "monkey"
        test_img_folder (str): Complete test image folder path
        pred_js_folder (str): Complete JSON prediction results folder path
        pred_hm_folder (str): Complete heatmap prediction results folder path
        dataset_folder (str): Complete path to the dataset folder
    """
    test_X, test_names = images_to_tensor(test_img_folder)
    stages = model.predict(test_X)
    if num_stage > 1:
        tensors = stages[-1]
    else:
        tensors = stages

    predictions = []
    for tensor in tensors:
        pred = hm_tensor_to_lst(tensor)
        predictions.append(pred)

    create_heatmap_files(tensors, test_names, dataset_animal, output_dir=dataset_folder)
    create_output_files(predictions, test_names, dataset_animal, output_dir=dataset_folder)
    print("converted predictions to labels")
    create_labeled_img_folder(pred_js_folder, test_img_folder, dataset_animal, output_dir=dataset_folder)
    create_labeled_hm_img_folder(pred_hm_folder, test_img_folder, dataset_animal,  output_dir=dataset_folder)
    print("done")


def fast_flow_inference(fast_flow_model, dataset_animal, test_tensor_folder, test_img_folder,
                        pred_js_folder, pred_hm_folder, dataset_folder):
    """ Use trained optical flow model to make predictions on pre-processed test tensors

    Args:
        fast_flow_model (Keras model): Trained flow model for inference predictions
        dataset_animal (str): One of "mouse", "fruitfly", "zebrafish", "monkey"
        test_tensor_folder (str): Complete path to optical flow preprocessed tensors of test images.
        test_img_folder (str): Complete path to test image folder
        pred_js_folder (str): Complete path to JSON prediction results folder
        pred_hm_folder (str): Complete path to heatmap prediction results folder
        dataset_folder (str): Complete path to the dataset folder
    """
    test_X, test_names = get_range_tensor(test_tensor_folder)
    tensor_lst = fast_flow_model.predict(test_X)

    hm_lst = []
    predictions = []

    for tensor in tensor_lst:
        hm_tensor = tensor[0]
        hm_lst.append(hm_tensor)
        pred = hm_tensor_to_lst(hm_tensor)
        predictions.append(pred)

    create_heatmap_files(hm_lst, test_names, dataset_animal, output_dir=dataset_folder)
    create_output_files(predictions, test_names, dataset_animal, output_dir=dataset_folder)
    print("converted predictions to labels")
    create_labeled_img_folder(pred_js_folder, test_img_folder, dataset_animal, output_dir=dataset_folder)
    create_labeled_hm_img_folder(pred_hm_folder, test_img_folder, dataset_animal, output_dir=dataset_folder)
    print("done")


def print_pck(eval_out):
    """ Print PCK evaluation output

    Args:
        eval_out: PCK evaluation output
    """
    ordered_dict = collections.OrderedDict(eval_out)
    for label, result in ordered_dict.items():
        print(label, result)
