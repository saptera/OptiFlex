import sys
import datetime
from models.DeepPoseKit import stacked_densenet
from models.DeepLabCut import deeplabcut
from models.LEAP import LEAP
from models.FlexibleBaseline import deconv_resnet_model, deconv_resnet_model_small, deconv_resnet_model_reduced
from utils.eval_func import grp_eval_pck
from training.train_func import train_generator
from training.post_train import save_record, inference, get_joint_list, print_pck

"""This SCRIPT generates a folder of base model predictions for a given folder of images and a pre-trained base model
"""
""" Parameters:
      # Dataset Info:
        dataset_animal (str):  One of "mouse", "fruitfly", "zebrafish", "monkey"
        dataset_view (str): "sd" or "btw" when animal is mouse, otherwise ""
      # Model Info:
        model_name (str): One of "LEAP", "StackedDenseNet", "DeepLabCut", or "FlexibleBaseline"
        baseline_version (str): "Small", "Reduced", or "Standard" when model_name is "FlexibleBaseline", otherwise ""
      # Input Folder Info:
        train_img_folder (str): Complete train set image folder path
        train_lbl_folder (str): Complete train set label folder path
        ver_img_folder (str): Complete verification set image folder path
        ver_lbl_folder (str): Complete verification set label folder path
        test_img_folder (str): Complete test set image folder path
        test_lbl_folder (str): Complete test set label folder path
        frame_count (int): Number of unique frames in the train set
        max_val (float): Maximum value for the label heatmaps
      # Input Folder Info:
        dataset_folder (str): Complete dataset folder path
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
data_view = "sd"    # sd or btm only when animal is

# Model Info
model_name = "FlexibleBaseline"
baseline_version = ""  # baseline version choices: Small, Reduced, Standard

# Input Folder Info
train_img_folder = "../../dataset/trn_minset_" + dataset_animal + "/img/" + data_view
train_lbl_folder = "../../dataset/trn_minset_" + dataset_animal + "/lbl/" + data_view
ver_img_folder = "../../dataset/vld_set_" + dataset_animal + "/img/" + data_view
ver_lbl_folder = "../../dataset/vld_set_" + dataset_animal + "/lbl/" + data_view
test_img_folder = "../../dataset/tst_set_" + dataset_animal + "/img/" + data_view
test_lbl_folder = "../../dataset/tst_set_" + dataset_animal + "/lbl/" + data_view
frame_count = 10
max_val = 16

# Output Folder Info
dataset_folder = "../../dataset/"
pred_js_folder = dataset_folder + "pred_js_lbl"
pred_hm_folder = dataset_folder + "pred_hm_lbl"

# Training Hyperparameters
batch_size = 10
epochs = 4
learning_rate = 0.0001
# -------------------------------------------------------------------------------------------------------------------- #

cur_time = datetime.datetime.now().strftime("%I:%M%p-%B-%d-%Y")
dataset_name = dataset_animal   # only changed when the animal is mouse
joint_list = get_joint_list(dataset_animal)


# configuration based on dataset
if dataset_animal == "mouse":

    dataset_name = dataset_name + "_" + data_view
    num_joints = 8
    img_width = 512
    img_height = 128

elif dataset_animal == "monkey":

    num_joints = 7
    img_width = 512
    img_height = 256

elif dataset_animal == "fruitfly":

    num_joints = 32
    img_width = 256
    img_height = 256

elif dataset_animal == "zebrafish":

    num_joints = 12
    img_width = 512
    img_height = 256

else:
    print("Dataset for " + dataset_animal + " is not currently available.")
    sys.exit("Please choose dataset animal from mouse, monkey, fruitfly, or zebrafish")

# configuration based on model
if model_name == "StackedDenseNet":

    num_stage = 3
    model = stacked_densenet(img_width, img_height, num_joints, learning_rate, 2)


elif model_name == "DeepLabCut":

    num_stage = 2
    model = deeplabcut(img_width, img_height, num_joints, learning_rate)

elif model_name == "LEAP":

    num_stage = 1
    model = LEAP(img_width, img_height, num_joints, learning_rate)

elif model_name == "FlexibleBaseline":

    model_name = baseline_version + model_name

    if baseline_version == "Standard":
        num_stage = 2
        model = deconv_resnet_model(img_width, img_height, num_joints, learning_rate)
    elif baseline_version == "Reduced":
        num_stage = 2
        model = deconv_resnet_model_reduced(img_width, img_height, num_joints, learning_rate)
    elif baseline_version == "Small":
        num_stage = 1
        model = deconv_resnet_model_small(img_width, img_height, num_joints, learning_rate)
    else:
        sys.exit(baseline_version+" not supported")

else:
    print(model_name + " is not currently a supported model.")
    sys.exit("We only support LEAP, StackedDenseNet, DeepLabCut, or FlexibleBaseline")

model_filename = "_".join([model_name, dataset_name, str(frame_count)+"frame", str(epochs)+"ep",
                           str(learning_rate)+"lr", str(batch_size)+"bs", cur_time]) + ".model"

# train model
model, history = train_generator(train_img_folder, train_lbl_folder, ver_img_folder, ver_lbl_folder,
                                 model, num_stage, batch_size, epochs, max_val)

# save
save_record(model, history, model_filename)

# inference
inference(model, num_stage, dataset_animal, test_img_folder, pred_js_folder, pred_hm_folder, dataset_folder)

# PCK
results = grp_eval_pck(pred_js_folder, test_lbl_folder, joint_list)
print_pck(results)
