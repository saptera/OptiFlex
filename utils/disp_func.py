import os
import pickle as pkl
import cv2
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
from data.lbl_proc import jsl_read, jsl_plot, hml_read, hml_plot, hml_jntplot, lbl_merge

"""Function list:
img_jsl_plt(img_file, jsl_file, color_list, annotate):  Create a labelled image for manual verification.
img_hml_plt(img_file, hml_file, joint, color_list):  Create a HeatMap labelled image for manual verification.
show_trace(jsl_src_dir, lbl_list, img_size, color_list):  Plot image trace results.
disp_model_hist(hist_file, hist_name):  Display model training history graph.
"""


def img_jsl_plt(img_file, jsl_file=None, color_list=None, annotate=True):
    """ Create a labelled image for manual verification.

    Args:
        img_file (str): Image file has been labelled.
        jsl_file (str or None): Labelling file related with image file (default: None = *.json file-pair [img_file]).
        color_list (dict[str, tuple[int, int, int]] or None): Dictionary of colors linked with each label.
        annotate (bool): Defines if label texts will be on the converted image.

    Returns:
        tuple[np.ndarray, str]:
            img (np.ndarray): {3D} Image array, image with labelling information.
            title (str): Name of the image.
    """
    # Get image file name string
    title = os.path.split(img_file)[1]    # Get image file name with extension
    title = os.path.splitext(title)[0]    # Remove extension

    # Verify defined label file related with current image
    if jsl_file is None:    # Check empty "lbl_file " input
        jsl_file = os.path.splitext(img_file)[0] + ".json"
    elif not os.path.isfile(jsl_file):    # Check if defined "lbl_file" is exist
        print(title + ": Label file not found, try default value!")
        jsl_file = os.path.splitext(img_file)[0] + ".json"
    # Load label file
    if not os.path.isfile(jsl_file):    # Check again if "lbl_file" is still missing
        print(title + ": No related label file found!")
        lbl_data = []
        flag = False    # Set status for plotting
    else:
        lbl_data = jsl_read(jsl_file)    # Import label data
        flag = True    # Set status for plotting

    # Plotting section
    img = cv2.imread(img_file, 1)    # cv::ImreadModes - enum 1, cv2.IMREAD_COLOR
    # Plot all labels to image
    if flag:
        img = jsl_plot(jsl_data=lbl_data, img=img, color_list=color_list, annotate=annotate)
    else:
        cv2.putText(img=img, text="No labels found!", org=(15, 35), fontFace=0, fontScale=1, color=(0, 0, 255))
        # cv2.putText::fontFace=0: FONT_HERSHEY_SIMPLEX
    return img, title


def img_hml_plt(img_file, hml_file=None, joint=None, color_list=None):
    """ Create a HeatMap labelled image for manual verification.

    Args:
        img_file (str): Image file has been labelled.
        hml_file (str or None): HeatMap label file related with image (default: None = *.pkl file-pair [img_file]).
        joint (str or list[str] or None): Only plot defined label if defined (default: None = plot ALL).
        color_list (dict[str, tuple[int, int, int]] or None): Dictionary of colors linked with label (default: None).

    Returns:
        tuple[np.ndarray, str]:
            img (np.ndarray): Image array, image with labelling information.
            title (str): Name of the image.
    """
    # Get image file name string
    title = os.path.split(img_file)[1]    # Get image file name with extension
    title = os.path.splitext(title)[0]    # Remove extension

    # Verify defined label file related with current image
    if hml_file is None:    # Check empty "heatlbl_file " input
        hml_file = os.path.splitext(img_file)[0] + ".pkl"
    elif not os.path.isfile(hml_file):    # Check if defined "heatlbl_file" is exist
        print(title + ": Label file not found, try default value!")
        hml_file = os.path.splitext(img_file)[0] + ".pkl"
    # Load label file
    if not os.path.isfile(hml_file):    # Check again if "heatlbl_file" is still missing
        print(title + ": No related label file found!")
        heatlbl_data = []
        flag = False    # Set status for plotting
    else:
        heatlbl_data = hml_read(hml_file)    # Import label data
        flag = True    # Set status for plotting

    # Plotting section
    img = cv2.imread(img_file, 1)    # cv::ImreadModes - enum 1, cv2.IMREAD_COLOR
    # Plot all labels to image
    if flag:
        if joint is None:
            img = hml_plot(hml_data=heatlbl_data, img=img, color_list=color_list)
        else:
            img = hml_jntplot(hml_data=heatlbl_data, img=img, joint=joint, color_list=color_list)
    else:
        cv2.putText(img=img, text="No labels found!", org=(15, 35), fontFace=0, fontScale=1, color=(0, 0, 255))
        # cv2.putText::fontFace=0: FONT_HERSHEY_SIMPLEX
    return img, title


def show_trace(jsl_src_dir, lbl_list, img_size=None, offset=0, color_list=None, smooth=False):
    """ Plot image trace results.

    Args:
        jsl_src_dir (str): Directory containing traced JSON label information.
        lbl_list (list[str]): List of label names to be plotted.
        img_size (tuple[int, int]): Original size of image being traced (default: None = calculate from data).
        offset (int): Define offset frame numbers (default: 0).
        color_list (list[str] or list[tuple] or None): Colors corresponding to label names (default: None = auto).
        smooth (bool): If the output should be smoothed (default = False).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            lbl_frm (np.ndarray): {1D} Array of frame numbers.
            lbl_crv_x (np.ndarray): Array of label LEFT positions, arranged with [lbl_list] order.
            lbl_crv_y (np.ndarray): Array of label TOP positions, arranged with [lbl_list] order.
    """
    # Read in data
    lbl_file = [os.path.join(jsl_src_dir, f) for f in os.listdir(jsl_src_dir) if f.endswith(".json")]
    frm_x, lbl_crv_x, lbl_crv_y = lbl_merge(lbl_file, lbl_list)
    # Set offset
    frm_x = frm_x - offset

    # Smoothing data
    if smooth:
        for n in range(len(lbl_list)):
            # Smoothing X
            s = CubicSpline(frm_x, lbl_crv_x[n], bc_type="natural", extrapolate=False)
            lbl_crv_x[n] = s(frm_x)
            # Smoothing Y
            s = CubicSpline(frm_x, lbl_crv_y[n], bc_type="natural", extrapolate=False)
            lbl_crv_y[n] = s(frm_x)

    # Plot data
    if img_size is None:
        img_size = (lbl_crv_x.max(), lbl_crv_y.max())
    # Plot X-axis results
    plt.figure("Image Trace Results at X-axis")
    plt.xlabel("Frame #")
    plt.ylabel("Rel. Pixel Loc.")
    for n in range(len(lbl_list)):
        if color_list is None:
            plt.plot(frm_x, (lbl_crv_x[n] + img_size[0] * n), linewidth=0.5, label=lbl_list[n])
        else:
            plt.plot(frm_x, (lbl_crv_x[n] + img_size[0] * n), color=color_list[n], linewidth=0.5, label=lbl_list[n])
    plt.axvline(0, color="gray")    # Show #0 frame location
    plt.legend()
    # Plot Y-axis results
    plt.figure("Image Trace Results at Y-axis")
    plt.xlabel("Frame #")
    plt.ylabel("Rel. Pixel Loc.")
    for n in range(len(lbl_list)):
        if color_list is None:
            plt.plot(frm_x, (lbl_crv_y[n] + img_size[1] * n), linewidth=0.5, label=lbl_list[n])
        else:
            plt.plot(frm_x, (lbl_crv_y[n] + img_size[1] * n), color=color_list[n], linewidth=0.5, label=lbl_list[n])
    plt.axvline(0, color="gray")  # Show #0 frame location
    plt.legend()
    # Display plot
    plt.show()

    # Output data for further usage
    return frm_x, lbl_crv_x, lbl_crv_y


def disp_model_hist(hist_file, hist_name=None):
    """Display model training history graph.

    Args:
        hist_file (str): File path of model training history (*.history).
        hist_name (str or None): Name of the plot.

    Returns:
    """
    # Read file and set plot
    hist_obj = pkl.load(open(hist_file, "rb"))
    fig = plt.figure(hist_name)
    ax = fig.gca()

    # Get data from history file
    trn_loss = hist_obj["loss"]
    vld_loss = hist_obj["val_loss"]
    x = [i + 1 for i in range(len(trn_loss))]
    # Plot data
    plt.plot(x, trn_loss, ls="-", c="xkcd:azure")
    plt.plot(x, vld_loss, ls="-", c="xkcd:orange")

    # Set X-axis properties
    plt.xticks([-1, len(trn_loss)])
    ax.xaxis.set_tick_params(which="major", size=4, width=0.5, direction="out")
    ax.xaxis.set_minor_locator(tkr.MultipleLocator(10))
    ax.xaxis.set_tick_params(which="minor", size=8, width=0.5, direction="out")
    plt.xlim(0, len(trn_loss) + 1)
    # Set Y-axis properties
    y_max = (int(max(trn_loss or vld_loss) * 100) // 5 + 1) * 5 / 100
    plt.yticks([0, y_max])
    ax.yaxis.set_tick_params(which="major", size=4, width=0.5, direction="out")
    ax.yaxis.set_minor_locator(tkr.LinearLocator(6))
    ax.yaxis.set_tick_params(which="minor", size=8, width=0.5, direction="out")
    plt.ylim(0, y_max)

    # Show plot
    plt.show()
