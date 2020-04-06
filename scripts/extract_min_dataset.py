import sys
import shutil
import random
from data.dataset_func import get_filenames


"""This SCRIPT extracts a dataset of smaller size from original dataset
"""
""" Parameters:
    img_folder (str):  Complete image folder path
    lbl_folder (str):  Complete lbl folder path
    out_img_folder (str):  Complete output image folder path for the extracted dataset
    out_lbl_folder (str):  Complete output label folder path for the extracted dataset
    num_img (int):  Number of non-augmented images to extract
"""

# Parameters input  -------------------------------------------------------------------------------------------------- #
img_folder = "../../dataset/trn_set/img"
lbl_folder = "../../dataset/trn_set/lbl"
out_img_folder = "../../dataset/trn_minset/img"
out_lbl_folder = "../../dataset/trn_minset/lbl"
num_img = 100
# -------------------------------------------------------------------------------------------------------------------- #

names = get_filenames(img_folder)
names = [name for name in names if name[0] != "."]

vid_code_lst = list(set([name[:13] for name in names]))

vid_frame_dict = {}
for code in vid_code_lst:
    vid_frame_dict[code] = []

for name in names:
    for code in vid_code_lst:
        if name.startswith(code):
            frame_code = name[18:]
            if frame_code not in vid_frame_dict[code]:
                vid_frame_dict[code].append(frame_code)

num_folder_img = sum([len(lst) for code, lst in vid_frame_dict.items()])

if num_img > num_folder_img:
    print("not enough images")
    sys.exit()

aug_code_lst = ["001", "002", "003", "004", "005", "006", "007", "008"]

# TODO: this assumes there is always enough images to choose from, every video code has at least one image
for i in range(num_img):
    code_idx = i % len(vid_code_lst)
    vid_code = vid_code_lst[code_idx]
    frame_code_lst = vid_frame_dict[vid_code]
    random.shuffle(frame_code_lst)
    frame_code = frame_code_lst.pop()
    if not vid_frame_dict[vid_code]:
        vid_code_lst.remove(vid_code)

    for aug_code in aug_code_lst:
        img_name = "_".join([vid_code, aug_code, frame_code])
        img_path = img_folder + img_name + ".png"
        lbl_path = lbl_folder + img_name + ".pkl"
        shutil.copy2(img_path, out_img_folder)
        # print("Copied", img_path, "to", out_img_folder)
        shutil.copy2(lbl_path, out_lbl_folder)
        # print("Copied", lbl_path, "to", out_lbl_folder)
