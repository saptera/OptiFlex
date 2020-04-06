import os
import cv2
from OptiFlex.utils.base_func import x64_sys
from OptiFlex.utils.disp_func import img_jsl_plt

"""This SCRIPT automatically create a display window for images in defined directory with labels plotted on.
"""
""" Parameters:
        img_path: {STR} Folder with image files.
        lbl_path: {STR} Folder with corresponding label files, use the same as [img_path] if left blank.
        color_list: {DICT} Dictionary of colors linked with each label, use random color if set to None.
        annotate: {BOOL} Defines if label texts will be on the converted image.
        prefix: {STR} Prefix of target files.
        extension: {STR} File extension of target image files.
"""

# Parameters input  -------------------------------------------------------------------------------------------------- #
img_path = "./data_set/"
lbl_path = ""
color_list = None
annotate = True
prefix = ""
extension = ".png"
# -------------------------------------------------------------------------------------------------------------------- #


# Standardize file input path
if not img_path.endswith("/"):
    img_path += "/"
if (lbl_path == str()) or (lbl_path is None):
    lbl_path = img_path
elif not lbl_path.endswith("/"):
    lbl_path += "/"

# Get all images in the folder
files = os.listdir(img_path)
img_lst = []
for img_file in files:
    if img_file.startswith(prefix) and img_file.endswith(extension):
        img_lst.append(img_file)
img_lst.sort()
frm_num = len(img_lst)

# Showing labelled images loop
i = 0
while True:
    # Create labelled image for display
    frm = img_lst[i]
    lbl = os.path.splitext(frm)[0] + ".json"
    title = frm[-9:-4]
    img, _ = img_jsl_plt(img_path + frm, lbl_path + lbl, color_list=color_list, annotate=annotate)

    # Displaying merged images with labels
    print("Current frame: " + title)
    cv2.imshow("Labelled Image", img)
    # System x86/x64 bit detection
    if x64_sys():
        k = cv2.waitKey(0) & 0xFF
    else:
        k = cv2.waitKey(0)
    # Keyboard control
    if k == ord("d"):    # Press "D" key for next frame
        if i < frm_num - 1:
            i += 1
        else:    # Loop back to the first image
            print("Last frame, loop back to first frame!")
            i = 0
        continue
    elif k == ord("a"):    # Press "A" key for previous frame
        if i > 0:
            i -= 1
        else:    # Loop back to the last image
            print("First frame, loop back to last frame!")
            i = frm_num - 1
        continue
    elif k == 27:    # Press "ESC" key to quit
        cv2.destroyWindow("Labelled Image")
        print("Exit system.")
        exit()
