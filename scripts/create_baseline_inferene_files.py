import os
from keras.models import load_model
from OptiFlex.data.lbl_proc import hml_write
from OptiFlex.data.dataset_func import images_to_tensor
from OptiFlex.training.post_train import get_joint_list, tensor_to_heatmap_file

"""This SCRIPT generates a folder of base model predictions for a given folder of images and a pre-trained base model
"""
""" Parameters:
      # Dataset Info
        dataset_animal (str):  One of "mouse", "fruitfly", "zebrafish", "monkey"
        dataset_view (str): "sd" or "btw" when animal is mouse, otherwise ""
        dataset_folder (str):  Complete dataset folder path
        dataset_type (str): "trn" for train set, "vrn" for verification set, and "tst" for test set
        img_folder (str):  Complete image folder path
      # Base Model Info
        base_model_path (str): Complete path to save Keras model file
        model_name (str): One of "LEAP", "StackedDenseNet", "DeepLabCut", or "FlexibleBaseline"
        num_stage (int): Number of stage for the base model
"""

# Parameters input  -------------------------------------------------------------------------------------------------- #
# Dataset Info
dataset_animal = "mouse"
dataset_view = "sd"
dataset_folder = "../../dataset/"
dataset_type = "trn"
img_folder = "../../dataset/trn_set_" + dataset_animal + "_seq" + "/img/" + dataset_view
# Base Model Info
base_model_path = "../../model/" + "SimpleBaseline_mouse_sd_epoch50_01:34AM-January-06-2020.model"
model_name = "FlexibleBaseline"
num_stage = 2
# -------------------------------------------------------------------------------------------------------------------- #


if __name__ == "__main__":
    inf_folder_name = "_".join([dataset_type, model_name, "inference"])
    inf_path = os.path.join(dataset_folder, inf_folder_name)

    try:
        os.mkdir(inf_path)
    except OSError:
        print("Creation of the directory %s failed" % inf_path)
    else:
        print("Successfully created the directory %s " % inf_path)

    img_tensor, img_names = images_to_tensor(img_folder)  # img_name are purely names without ending or path info
    base_model = load_model(base_model_path)
    base_model._make_predict_function()

    stages = base_model.predict(img_tensor)
    # we get base_tensors that that are hm tensor predictions from base model
    if num_stage > 1:
        base_tensors = stages[-1]
    else:
        base_tensors = stages

    num_img = len(img_names)
    joint_list = get_joint_list(dataset_animal)
    for i in range(num_img):
        hm_file = tensor_to_heatmap_file(base_tensors[i], joint_list)
        hml_write(os.path.join(inf_path, img_names[i] + '.pkl'), hm_file)
