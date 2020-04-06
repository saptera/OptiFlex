import os
import copy
import numpy as np
import cv2
from OptiFlex.data.lbl_proc import hml_read, hm_optical_warp, hml_plot

"""This SCRIPT helps to pre-check the quality of OpticalFlow parameters.
"""
""" Parameters:
        img_dir: {STR} Folder with image files.
        hml_dir: {STR} Folder with corresponding HeatMap label files, use the same as [img_dir] if left blank.
        step: {INT} Define the OpticalFlow frame step size.
        color_list: {DICT} Dictionary of colors linked with each label.
"""

# Parameters input  -------------------------------------------------------------------------------------------------- #
img_dir = "./img/"
hml_dir = "./lbl/"
step = 2
color_list = {}
# -------------------------------------------------------------------------------------------------------------------- #


# Get required file list
img_lst = []    # INIT VAR
hml_lst = []    # INIT VAR
hml_dir = img_dir if hml_dir is None or hml_dir == str() else hml_dir
for f in os.listdir(img_dir):
    if f.endswith(".png"):
        img_lst.append(os.path.join(img_dir, f))
        hml_lst.append(os.path.join(hml_dir, os.path.splitext(f)[0] + ".pkl"))


def hml_warp(img_ref, img_tgt, hml_data):
    # Convert HeatMap label data
    hm_label = []
    hm_stack = []
    for hm in hml_data:
        hm_label.append(hm["label"])
        hm_stack.append(hm["heatmap"])
    # Perform OpticalFlow and convert data
    hm_tensor = np.stack(hm_stack, axis=-1)[:, :, :, None]
    hm_warp = hm_optical_warp(img_ref, img_tgt, hm_tensor)
    hml_optf = []
    for n in range(len(hm_label)):
        hm_temp = {"label": hm_label[n], "heatmap": hm_warp[:, :, n]}
        hml_optf.append(copy.deepcopy(hm_temp))
    return hml_optf


def optical_warp_check(tgt, rng):
    # Read images
    img_prev = cv2.imread(img_lst[tgt - rng], cv2.IMREAD_UNCHANGED)
    img_tgt = cv2.imread(img_lst[tgt], cv2.IMREAD_UNCHANGED)
    img_next = cv2.imread(img_lst[tgt + rng], cv2.IMREAD_UNCHANGED)
    # Read labels
    hml_prev = hml_read(hml_lst[tgt - rng])
    hml_tgt = hml_read(hml_lst[tgt])
    hml_next = hml_read(hml_lst[tgt + rng])

    # Perform OpticalFlow
    wrp_prev = hml_warp(img_prev, img_tgt, hml_prev)
    wrp_next = hml_warp(img_next, img_tgt, hml_next)

    # Plotting PREV_IMG
    org_prev = hml_plot(hml_prev, img_prev, color_list)
    cv2.putText(img=org_prev, text="Image: PREV", org=(15, 15), fontFace=0, fontScale=0.5, color=(0, 255, 0))
    # Plotting PREV_WRP
    plt_prev = hml_plot(wrp_prev, img_tgt, color_list)
    cv2.putText(img=plt_prev, text="Flow: PREV -> TGT", org=(15, 15), fontFace=0, fontScale=0.5, color=(0, 255, 0))
    # Plotting TGT_IMG
    org_tgt = hml_plot(hml_tgt, img_tgt, color_list)
    cv2.putText(img=org_tgt, text="Image: TGT", org=(15, 15), fontFace=0, fontScale=0.5, color=(255, 255, 255))
    # Plotting NEXT_WRP
    plt_next = hml_plot(wrp_next, img_tgt, color_list)
    cv2.putText(img=plt_next, text="Flow: NEXT -> TGT", org=(15, 15), fontFace=0, fontScale=0.5, color=(0, 0, 255))
    # Plotting NEXT_IMG
    org_next = hml_plot(hml_next, img_next, color_list)
    cv2.putText(img=org_next, text="Image: NEXT", org=(15, 15), fontFace=0, fontScale=0.5, color=(0, 0, 255))
    # Stack images
    disp_img = np.vstack((org_prev, plt_prev, org_tgt, plt_next, org_next))

    return disp_img


tot_num = len(img_lst) - step - 1
# Showing labelled images loop
i = step
while True:
    # Create labelled image for display
    img = optical_warp_check(i, step)

    # Displaying merged images with labels
    print("Current flow: No.[%05d] -> No.[%05d] <- No.[%05d]" % (i - step, i, i + step))
    cv2.imshow("OpticalFlow of STEP-%d" % step, img)
    k = cv2.waitKey(0)
    # Keyboard control
    if k == ord("d"):    # Press "D" key for next frame
        if i < tot_num:
            i += 1
        else:    # Loop back to the first image
            print("Last frame, loop back to first frame!")
            i = step
        continue
    elif k == ord("a"):    # Press "A" key for previous frame
        if i > step:
            i -= 1
        else:    # Loop back to the last image
            print("First frame, loop back to last frame!")
            i = tot_num
        continue
    elif k == 27:    # Press "ESC" key to quit
        cv2.destroyAllWindows()
        print("Exit system.")
        exit()
