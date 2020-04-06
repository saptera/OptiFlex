import os
import pickle as pkl
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, MultipleLocator

"""This SCRIPT displays training and validation loss of models for comparison.
"""
""" Parameters:
        hist_dir: {STR} Folder with model history label files.
        dset_str: {STR} String to define which dataset to compare.
        loss_lst: {DICT} Dictionary of loss key linked with each model.
        color_lst: {DICT} Dictionary of colors linked with each model.
"""

# Parameters input  -------------------------------------------------------------------------------------------------- #
hist_dir = "./model/"
dset_str = ""
loss_lst = {"DeepLabCut": "conv2d_transpose_2_loss", "FlexibleBaseline": "conv2d_1_loss",
            "LEAP": "loss", "StackedDenseNet": "conv2d_88_loss"}
color_lst = {"FlexibleBaseline": "#1DA800", "DeepLabCut": "#FF0300", "LEAP": "#1F3CFF", "StackedDenseNet": "#FF6300"}
# -------------------------------------------------------------------------------------------------------------------- #


def read_model_history(history_file_path):
    history = pkl.load(open(history_file_path, "rb"))
    return history


def plot_setup():
    # Setup plot spines
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    # Setup X-axis
    mpl.rcParams['xtick.major.size'] = 4
    mpl.rcParams['xtick.major.width'] = 0.5
    mpl.rcParams['xtick.minor.size'] = 8
    mpl.rcParams['xtick.minor.width'] = 0.5
    # Setup Y-axis
    mpl.rcParams['ytick.major.size'] = 4
    mpl.rcParams['ytick.major.width'] = 0.5
    mpl.rcParams['ytick.minor.size'] = 8
    mpl.rcParams['ytick.minor.width'] = 0.5


# Get file list
hist_lst = []
name_lst = []
for f in os.listdir(hist_dir):
    if f.endswith(".history"):
        hist_lst.append(os.path.join(hist_dir, f))
        name_lst.append(f.partition("_")[0])
# Read required data
trn_loss_lst = []
vld_loss_lst = []
clr_name_lst = []
for i in range(len(hist_lst)):
    if dset_str in hist_lst[i]:
        model_history = read_model_history(hist_lst[i])
        key = name_lst[i]
        trn_loss_lst.append(model_history[loss_lst[key]])
        vld_loss_lst.append(model_history["val_" + loss_lst[key]])
        clr_name_lst.append(key)


# Plot setup
plot_setup()
x = [i + 1 for i in range(len(max(trn_loss_lst, key=len)))]

# Plot for training loss results
fig = plt.figure("Model Training Loss of Dataset [%s]" % dset_str)
ax = fig.gca()
for i in range(len(trn_loss_lst)):
    plt.plot(x, trn_loss_lst[i], c=color_lst[clr_name_lst[i]])
# Set X-axis properties
plt.xticks([-1, len(x)])
ax.xaxis.set_minor_locator(MultipleLocator(10))
plt.xlim(0, len(x) + 1)
# Set Y-axis properties
y_max = (int(max([max(l) for l in trn_loss_lst]) * 100) // 5 + 1) * 5 / 100
plt.yticks([0, y_max])
ax.yaxis.set_minor_locator(LinearLocator(6))
plt.ylim(0, y_max)

# Plot for validation loss results
fig = plt.figure("Model Validation Loss of Dataset [%s]" % dset_str)
ax = fig.gca()
for i in range(len(vld_loss_lst)):
    plt.plot(x, vld_loss_lst[i], c=color_lst[clr_name_lst[i]])
# Set X-axis properties
plt.xticks([-1, len(x)])
ax.xaxis.set_minor_locator(MultipleLocator(10))
plt.xlim(0, len(x) + 1)
# Set Y-axis properties
y_max = (int(max([max(l) for l in vld_loss_lst]) * 100) // 5 + 1) * 5 / 100
plt.yticks([0, y_max])
ax.yaxis.set_minor_locator(LinearLocator(6))
plt.ylim(0, y_max)

# Show plot
plt.show()
