import random
import numpy as np
from keras.models import load_model
from timeit import default_timer as timer
from data.dataset_func import images_to_tensor

"""This SCRIPT measures the inference speed of LEAP, DeepLabCut, FlexibleBaseline, and StackedDenseNet
"""
""" Parameters:
      # Dataset Info
        dataset_animal (str):  One of "mouse", "fruitfly", "zebrafish", "monkey"
        dataset_view (str): "sd" or "btw" when animal is mouse, otherwise ""
        test_img_folder (str): Complete test image folder path
      # Base Model Info
        dlc_model_path (str): Complete path to a saved DeepLabCut model file
        dpk_model_path (str): Complete path to a saved StackedDenseNet model file
        leap_model_path (str): Complete path to a saved LEAP model file
        flex_model_path (str): Complete path to a saved FlexibleBaseline model file        
"""

# Parameters input  -------------------------------------------------------------------------------------------------- #
# Dataset Info
dataset_animal = ""
data_view = ""
test_img_folder = "../../dataset/tst_set_" + dataset_animal + "/img/" + data_view
# Base Model Info
dlc_model_path = "../../model/" + ""
dpk_model_path = "../../model/" + ""
leap_model_path = "../../model/" + ""
flex_model_path = "../../model/" + ""
# -------------------------------------------------------------------------------------------------------------------- #

dlc_model = load_model(dlc_model_path)
dlc_model._make_predict_function()
dpk_model = load_model(dlc_model_path)
dpk_model._make_predict_function()
leap_model = load_model(dlc_model_path)
leap_model._make_predict_function()
flex_model = load_model(dlc_model_path)
flex_model._make_predict_function()

test_X, test_names = images_to_tensor(test_img_folder)
num_img = len(test_X)
print("Dataset: ", dataset_animal, data_view)
print("Number of Image: ", num_img)

models_names = [(dlc_model, "DeepLabCut"), (dpk_model, "DeepPoseKit"),
                (leap_model, "LEAP"), (flex_model, "FlexibleBaseline")]

dlc_times_128 = []
dpk_times_128 = []
leap_times_128 = []
flex_times_128 = []

dlc_times_1 = []
dpk_times_1 = []
leap_times_1 = []
flex_times_1 = []


for i in range(16):

    random.shuffle(models_names)
    for model, name in models_names:
        print(name)
        start_time_128 = timer()
        _ = model.predict(test_X, batch_size=128)
        time_taken_128 = timer() - start_time_128

        start_time_1 = timer()
        _ = model.predict(test_X, batch_size=1)
        time_taken_1 = timer() - start_time_1

        if name == "DeepLabCut":
            dlc_times_128.append(time_taken_128)
            dlc_times_1.append(time_taken_1)
        elif name == "DeepPoseKit":
            dpk_times_128.append(time_taken_128)
            dpk_times_1.append(time_taken_1)
        elif name == "LEAP":
            leap_times_128.append(time_taken_128)
            leap_times_1.append(time_taken_1)
        elif name == "FlexibleBaseline":
            flex_times_128.append(time_taken_128)
            flex_times_1.append(time_taken_1)
        else:
            print("Something is wrong!")


def print_results(base_model_name, times, num_image):
    img_times = [time / num_image for time in times]
    avg_time_taken = np.mean(times)
    std_time_taken = np.std(times)
    avg_img_time = np.mean(img_times) * 1000
    std_img_time = np.std(img_times) * 1000

    print(base_model_name + " Average Run-Time: %.8f s" % avg_time_taken)
    print(base_model_name + " Run-Time Standard Deviation: %.8f s" % std_time_taken)
    print(base_model_name + " Average Run-Time Per Image: %.8f ms" % avg_img_time)
    print(base_model_name + " Run-Time Per Image Standard Deviation: %.8f ms" % std_img_time)


model_name = "DeepLabCut"
print("Batch Size: ", 128)
print_results(model_name, dlc_times_128, num_img)
print("Batch Size: ", 1)
print_results(model_name, dlc_times_1, num_img)

model_name = "DeepPoseKit"
print("Batch Size: ", 128)
print_results(model_name, dpk_times_128, num_img)
print("Batch Size: ", 1)
print_results(model_name, dpk_times_1, num_img)

model_name = "LEAP"
print("Batch Size: ", 128)
print_results(model_name, leap_times_128, num_img)
print("Batch Size: ", 1)
print_results(model_name, leap_times_1, num_img)

model_name = "FlexibleBaseline"
print("Batch Size: ", 128)
print_results(model_name, flex_times_128, num_img)
print("Batch Size: ", 1)
print_results(model_name, flex_times_1, num_img)











