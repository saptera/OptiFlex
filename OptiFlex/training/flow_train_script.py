import sys
import datetime
from OptiFlex.models.OpticalFlow import flow_model
from OptiFlex.utils.eval_func import grp_eval_pck
from OptiFlex.training.train_func import train_fast_flow_generator
from OptiFlex.training.post_train import save_record, fast_flow_inference, get_joint_list, print_pck

"""This SCRIPT generates a folder of base model predictions for a given folder of images and a pre-trained base model
"""
""" Parameters:
      # Dataset Info:
        dataset_animal (str):  One of "mouse", "fruitfly", "zebrafish", "monkey"
        dataset_view (str): "sd" or "btw" when animal is mouse, otherwise ""
      # Optical Flow Hyperparameters
        base_model_name (str): One of "LEAP", "StackedDenseNet", "DeepLabCut", or "FlexibleBaseline"
        frame_range (int): Number of adjacent frames to consider from before and after the reference frame
        skip_ratio (int): Sampling ratio for extracting adjacent frames. E.g. 1 means extract every frame, 2 means
        extract every two frames, etc.
      # Input Folder Info:
        dataset_folder (str): Complete dataset folder path
        tensor_folder_name (str): Name of the folder that contains optical flow preprocessed tensor 
        train_tensor_folder (str): Complete train set preprocessed tensor folder path
        train_lbl_folder (str): Complete train set label folder path
        ver_tensor_folder (str): Complete verification set preprocessed tensor folder path
        ver_lbl_folder (str): Complete verification set label folder path
        test_tensor_folder (str): Complete test set preprocessed tensor folder path
        test_img_folder (str): Complete test set image folder path
        test_lbl_folder (str): Complete test set label folder path
        frame_count (int): Number of unique frames in the train set
        max_val (float): Maximum value for the label heatmaps
      # Input Folder Info:
        pred_js_folder (str): Complete path to JSON prediction results folder
        pred_hm_folder (str): Complete path to heatmap prediction results folder
      # Training Hyperparameters:
        batch_size (int): Number of image-label pairs to be included in each training batch
        epochs (int): Number of time the entire dataset will be passed through during training
        learning_rate (float): Scalar value indicating speed of gradient update
"""

# Parameters input  -------------------------------------------------------------------------------------------------- #
# Dataset Info
dataset_animal = "mouse"
data_view = "sd"

# Optical Flow Hyperparameters
base_model_name = ""
skip_ratio = 1
frame_range = 4

# Input Folder Info
dataset_folder = "../../dataset/"
tensor_folder_name = "_".join([base_model_name, "s"+str(skip_ratio), "f"+str(frame_range), "tensor"])
train_tensor_folder = dataset_folder + "trn_tensor_" + dataset_animal + "_" + data_view + "/" + tensor_folder_name
train_lbl_folder = dataset_folder + "trn_tensor_" + dataset_animal + "_" + data_view + "/lbl/"
ver_tensor_folder = dataset_folder + "ver_tensor_" + dataset_animal + "_" + data_view + "/" + tensor_folder_name
ver_lbl_folder = dataset_folder + "ver_tensor_" + dataset_animal + "_" + data_view + "/lbl/"
test_tensor_folder = dataset_folder + "tst_tensor_" + dataset_animal + "_" + data_view + "/" + tensor_folder_name
test_img_folder = dataset_folder + "tst_set_" + dataset_animal + "_seq" + "/img/" + data_view
test_lbl_folder = dataset_folder + "tst_set_" + dataset_animal + "_seq" + "/lbl/" + data_view
max_val = 16

# Output Folder Info
pred_js_folder = dataset_folder + "pred_js_lbl"
pred_hm_folder = dataset_folder + "pred_hm_lbl"

# Training Hyperparameters
batch_size = 10
epochs = 4
learning_rate = 0.0001
# -------------------------------------------------------------------------------------------------------------------- #

cur_time = datetime.datetime.now().strftime("%I:%M%p-%B-%d-%Y")
joint_list = get_joint_list(dataset_animal)

# configuration based on dataset
if dataset_animal == "mouse":

    dataset_name = dataset_animal + "_" + data_view
    num_joints = 8
    img_width = 512
    img_height = 128

else:
    sys.exit("Animal Not Supported")

model_filename = "_".join(["OpticalFlow", base_model_name, dataset_name, str(epochs)+"ep", str(learning_rate)+"lr",
                           str(batch_size)+"bs", cur_time]) + ".model"

# Get Untrained OpticalFlow Model
flow_model = flow_model(img_width, img_height, num_joints, learning_rate, frame_range)

# Train OpticalFlow Model
model, history = train_fast_flow_generator(train_tensor_folder, train_lbl_folder, ver_tensor_folder, ver_lbl_folder,
                                           flow_model, batch_size, epochs, max_val)

# Save
save_record(model, history, model_filename)

# OpticalFlow Inference
fast_flow_inference(model, dataset_animal, test_tensor_folder, test_img_folder,
                    pred_js_folder, pred_hm_folder, dataset_folder)

# PCK results
pck = grp_eval_pck(pred_js_folder, test_lbl_folder, joint_list)
print_pck(pck)
