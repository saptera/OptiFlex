import os
import multiprocessing as mp
import numpy as np
import cv2
from OptiFlex.utils.base_func import os_rand_range, make_outdirs, read_pathlist_file
from OptiFlex.data.img_proc import imhml_resize, imhml_flip, imhml_rotate
from OptiFlex.data.lbl_proc import hml_read, hml_write, hml_verify

"""This SCRIPT creates augmented dataset(s) with corresponding HeatMap label file by random flipping and/or rotating.
   Different from [create_augmented_heatmap_set.py] this SCRIPT lock the flipping and rotating within one set.
   Specially defined CSV files will be used in this script, whose 1st-col is set name, 2nd-col is set path.
   All image will be resized to same size to simplify machine learning process.
"""
""" Parameters:
      # Basic settings
        trn_lst: {STR} A CSV file defining dataset for training, no set will be generated if this value is None.
        aug_nr: {INT} Defines how many set(s) will be generated, no set will be generated if this value <= 0.
        vld_lst: {STR} A CSV file defining dataset for validation, no set will be generated if this value is None.
        tst_lst: {STR} A CSV file defining dataset for testing, no set will be generated if this value is None.
        group_list: {LIST of STR} Names used to separate image/label into different groups, set to None if no groups.
        out_dir: {STR} Folder for outputting generated files.
      # Image transform settings
        size: {TUPLE of 2-INT} Defines final size of image (in pixel), sizes in power of 2 are preferred:
                size[0] = width    ||    size[1] = height
        rnd_flp {BOOL}: Control if the random flipping will be used in augmentation.
        axis: {LIST of -1 &/| 0 &/| 1} Specify how to flip the array (randomized in choice):
                1 = y-axis    ||    0 = x-axis    ||    -1 = both axes
        rnd_rot {BOOL}: Control if the random rotation will be used in augmentation.
        angle: {TUPLE of 2-FLOAT} Defines rotation angle of image (randomized in range).
                > 0 = CCW    ||    < 0 = CW
        peak {INT or FLOAT}: Peak value of transformed HeatMap.
      # Label verification settings
        lbl_std: {LIST of STR} List of all label tags used in labelling, including duplicate, order sensitive.
        hml_dummy: {2D-ARRAY} HeatMap used to override when some label is missing.
        keep_dum: {BOOL} Defines if a image/label pair with dummy data inserted will be saved.
        force_tst: {Bool} Always save image/label pair within test sets, ignoring [keep_dum] setting.
      # Parallel processing settings
        workers: {INT} Number of logical cores to be used in pool.
"""

# Parameters input  -------------------------------------------------------------------------------------------------- #
# Basic settings
trn_lst = "./lst_trn.csv"
aug_nr = 8
vld_lst = "./lst_vld.csv"
tst_lst = "./lst_tst.csv"
group_list = []
out_dir = "./data_set"
# Image transformation settings
size = (256, 256)
rnd_flp = True
axis = [-1, 0, 1]
rnd_rot = True
angle = (-10, 10)
peak = 16.0
# Label verification settings
lbl_std = []
hml_dummy = None
keep_dum = False
force_tst = True
# Parallel processing settings
workers = 8
# -------------------------------------------------------------------------------------------------------------------- #


# Define a dataset folder scan function
def dat_dir_scan(file_path):
    img_list = []    # INIT VAR
    lbl_list = []    # INIT VAR
    grp_list = []    # INIT VAR
    n_list = []    # INIT VAR
    for fp in file_path:
        if os.path.isdir(fp):
            files = os.listdir(fp)  # Get files in folder
            img_list_temp = []    # INIT/RESET VAR
            lbl_list_temp = []    # INIT/RESET VAR
            grp_list_temp = []    # INIT/RESET VAR
            n_list_temp = []    # INIT/RESET VAR
            for f in files:
                if f.endswith(".png"):
                    img_list_temp.append(f)
                    lbl_list_temp.append(os.path.splitext(f)[0] + ".pkl")    # Switch extension
                    # Set group
                    if group_list is None or group_list == []:
                        grp_list_temp.append(0)
                    else:
                        grp_idx = 0
                        for g in group_list:
                            if f.startswith(g):
                                grp_list_temp.append(grp_idx)
                                break
                            else:
                                grp_idx += 1
                n_list_temp = list(range(len(img_list_temp)))    # Creat a iterable list for multi-processing
            # Passing temp list data to the main list
            img_list.append(img_list_temp)
            lbl_list.append(lbl_list_temp)
            grp_list.append(grp_list_temp)
            n_list.append(n_list_temp)
        else:
            print("    Invalid dataset folder [%s]!" % fp)
    return img_list, lbl_list, grp_list, n_list


# Define a dataset output folder creation function
def dat_set_out(set_name):
    # Set main output path
    set_img_dir = make_outdirs(os.path.join(out_dir, (set_name + "/img/")), err_msg="    Invalid dataset image path!")
    set_lbl_dir = make_outdirs(os.path.join(out_dir, (set_name + "/lbl/")), err_msg="    Invalid dataset label path!")
    # Set group output path
    grp_img_dir = []    # INIT VAR
    grp_lbl_dir = []    # INIT VAR
    if group_list is None or group_list == []:
        grp_img_dir.append(set_img_dir)
        grp_lbl_dir.append(set_lbl_dir)
    else:
        for g in group_list:
            img_dir_temp = make_outdirs(set_img_dir + g + "/",
                                        err_msg=("    Invalid dataset image path for group [%s]!" % g))
            lbl_dir_temp = make_outdirs(set_lbl_dir + g + "/",
                                        err_msg=("    Invalid dataset label path for group [%s]!" % g))
            grp_img_dir.append(img_dir_temp)
            grp_lbl_dir.append(lbl_dir_temp)
    return grp_img_dir, grp_lbl_dir


# Define a random transform parameters generator
def rand_trans_gen():
    # Get random session
    flip = np.random.choice([True, False])
    rotate = np.random.choice([True, False])
    # Get random parameters
    gen_axis = np.random.choice(axis) if (rnd_flp and flip) else None
    gen_angle = os_rand_range(angle[0], angle[1], size=8, digits=2) if (rnd_rot and rotate) else None
    return gen_axis, gen_angle


# Define data augmentation function to be parallelled
def imhml_seqaug_set(nr, main_index, img_flst, lbl_flst, dir_lst, set_nlst, set_nr, rand_axis, rand_angle,
                     grp_dlst, img_grp_dir, lbl_grp_dir):
    img_file = img_flst[main_index][nr]
    lbl_file = lbl_flst[main_index][nr]
    # Read in files
    img = cv2.imread(os.path.join(dir_lst[main_index], img_file), -1)    # cv::ImreadModes -1, cv2.IMREAD_UNCHANGED
    if os.path.isfile(os.path.join(dir_lst[main_index], lbl_file)):    # Check if some label file is missing
        lbl = hml_read(os.path.join(dir_lst[main_index], lbl_file))
    else:
        lbl = []
    # Process augmentation
    if rand_axis is not None:
        img, lbl = imhml_flip(img, lbl, rand_axis, peak)    # Flip
    if rand_angle is not None:
        img, lbl = imhml_rotate(img, lbl, rand_angle, peak)    # Rotate
    img, lbl = imhml_resize(img, lbl, size, 3, peak)    # interpolation = 3: INTER_AREA
    # Standardize label output
    lbl, has_dum = hml_verify(lbl, lbl_std, hml_dummy)
    # Output files
    if keep_dum or (not has_dum):    # Converse implication
        # Format output file name
        img_out_file = "%s_%0.3d_%s" % (set_nlst[main_index], set_nr, img_file)
        lbl_out_file = "%s_%0.3d_%s" % (set_nlst[main_index], set_nr, lbl_file)
        # Save by group
        grp_def = grp_dlst[main_index][nr]
        cv2.imwrite(os.path.join(img_grp_dir[grp_def], img_out_file), img)
        hml_write(os.path.join(lbl_grp_dir[grp_def], lbl_out_file), lbl)


# Define validation set creation function to be parallelled
def imhml_seqvld_set(nr, main_index, img_flst, lbl_flst, dir_lst, set_nlst, grp_dlst, img_grp_dir, lbl_grp_dir):
    img_file = img_flst[main_index][nr]
    lbl_file = lbl_flst[main_index][nr]
    # Read in files
    img = cv2.imread(os.path.join(dir_lst[main_index], img_file), -1)    # cv::ImreadModes -1, cv2.IMREAD_UNCHANGED
    if os.path.isfile(os.path.join(dir_lst[main_index], lbl_file)):    # Check if some label file is missing
        lbl = hml_read(os.path.join(dir_lst[main_index], lbl_file))
    else:
        lbl = []
    # Process resize only to images for validation sets
    img, lbl = imhml_resize(img, lbl, size, 3, peak)    # interpolation = 3: INTER_AREA
    # Standardize label output
    lbl, has_dum = hml_verify(lbl, lbl_std, hml_dummy)
    # Output files
    if keep_dum or (not has_dum):    # Converse implication
        # Format output file name
        img_out_file = "%s_vld_%s" % (set_nlst[main_index], img_file)
        lbl_out_file = "%s_vld_%s" % (set_nlst[main_index], lbl_file)
        # Save by group
        grp_def = grp_dlst[main_index][nr]
        cv2.imwrite(os.path.join(img_grp_dir[grp_def], img_out_file), img)
        hml_write(os.path.join(lbl_grp_dir[grp_def], lbl_out_file), lbl)


# Define testing set creation function to be parallelled
def imhml_seqtst_set(nr, main_index, img_flst, lbl_flst, dir_lst, set_nlst, grp_dlst, img_grp_dir, lbl_grp_dir):
    img_file = img_flst[main_index][nr]
    lbl_file = lbl_flst[main_index][nr]
    # Read in files
    img = cv2.imread(os.path.join(dir_lst[main_index], img_file), -1)    # cv::ImreadModes -1, cv2.IMREAD_UNCHANGED
    if os.path.isfile(os.path.join(dir_lst[main_index], lbl_file)):    # Check if some label file is missing
        lbl = hml_read(os.path.join(dir_lst[main_index], lbl_file))
    else:
        lbl = []
    # Process resize only to images for testing sets
    img, lbl = imhml_resize(img, lbl, size, 3, peak)    # interpolation = 3: INTER_AREA
    # Standardize label output
    lbl, has_dum = hml_verify(lbl, lbl_std, hml_dummy)
    # Output files
    grp_def = grp_dlst[main_index][nr]
    if force_tst or (keep_dum or (not has_dum)):  # Test set saving logic
        # Format output file name
        img_out_file = "%s_tst_%s" % (set_nlst[main_index], img_file)
        lbl_out_file = "%s_tst_%s" % (set_nlst[main_index], lbl_file)
        # Save by group
        cv2.imwrite(os.path.join(img_grp_dir[grp_def], img_out_file), img)
        hml_write(os.path.join(lbl_grp_dir[grp_def], lbl_out_file), lbl)


# Parallel processing session
if __name__ == '__main__':
    print("Process started!\n")

    # Pre-processing section -------------------------------------------------------------------------------------------
    print("Pre-processing:\n----------------------------------------------------------------")
    # Check training dataset definition
    if (trn_lst is None or trn_lst == str()) or (aug_nr is None or aug_nr <= 0):
        print("No dataset defined for training!")
        trn_set = False
    else:
        if os.path.isfile(trn_lst):
            print("Collecting information for training sets...")
            trn_dat_name, trn_dat_path = read_pathlist_file(trn_lst)    # Read CSV file for dataset list
            trn_img_list, trn_lbl_list, trn_grp_list, trn_n_list = dat_dir_scan(trn_dat_path)    # Get required files
            trn_grp_img_dir, trn_grp_lbl_dir = dat_set_out("trn_set")    # Set output folders
            print("    Success!")
            trn_set = True
        else:
            print("Invalid dataset list for training!")
            trn_set = False

    # Check validation dataset definition
    if vld_lst is None or vld_lst == str():
        print("No dataset defined for validation!")
        vld_set = False
    else:
        if os.path.isfile(vld_lst):
            print("Collecting information for validation sets...")
            vld_dat_name, vld_dat_path = read_pathlist_file(vld_lst)    # Read CSV file for dataset list
            vld_img_list, vld_lbl_list, vld_grp_list, vld_n_list = dat_dir_scan(vld_dat_path)    # Get required files
            vld_grp_img_dir, vld_grp_lbl_dir = dat_set_out("vld_set")    # Set output folders
            print("    Success!")
            vld_set = True
        else:
            print("Invalid dataset list for validation!")
            vld_set = False

    # Check testing dataset definition
    if tst_lst is None or tst_lst == str():
        print("No dataset defined for testing!")
        tst_set = False
    else:
        if os.path.isfile(tst_lst):
            print("Collecting information for testing sets...")
            tst_dat_name, tst_dat_path = read_pathlist_file(tst_lst)    # Read CSV file for dataset list
            tst_img_list, tst_lbl_list, tst_grp_list, tst_n_list = dat_dir_scan(tst_dat_path)    # Get required files
            tst_grp_img_dir, tst_grp_lbl_dir = dat_set_out("tst_set")    # Set output folders
            print("    Success!")
            tst_set = True
        else:
            print("Invalid dataset list for testing!")
            tst_set = False

    # Final check before processing
    if not (trn_set or vld_set or tst_set):
        print("Not a single dataset defined, system out!")
        exit()
    else:
        print("- - - - - - - - - - - - - - - -")

    # Generate all required random parameters
    ran_data = set()    # INIT VAR, use [set()] to avoid duplicated operation
    while len(ran_data) < aug_nr:
        ran_temp = rand_trans_gen()
        ran_data.add(ran_temp)
    ax = []    # INIT VAR
    ang = []    # INIT VAR
    for rd in ran_data:    # Get required values out from [set]
        ax.append(rd[0])
        ang.append(rd[1])

    # Setting up parallel process --------------------------------------------------------------------------------------
    print("Setup multiprocessing...")
    log_core = mp.cpu_count()
    if workers > log_core:
        print("    Not enough logic cores available as defined, using all available cores!")
        workers = log_core
    print("    Detected %d logic cores, connecting to %d workers.\n" % (log_core, workers))

    # Major processing loops -------------------------------------------------------------------------------------------
    # Training set creation
    if trn_set:
        print("Creating defined training sets:\n----------------------------------------------------------------")
        for i in range(len(trn_dat_path)):
            if i != 0:
                print("- - - - - - - - - - - - - - - -")
            print("Creating training sets for dataset [%s]:" % trn_dat_name[i])
            # Sets creation
            n = 1
            while n <= aug_nr:
                # Start parallel processing
                print("    Creating augmented dataset No.%0.3d..." % n)
                pool = mp.Pool(workers)
                for j in trn_n_list[i]:
                    pool.apply_async(imhml_seqaug_set, args=(j, i, trn_img_list, trn_lbl_list,
                                                             trn_dat_path, trn_dat_name, n, ax[n - 1], ang[n - 1],
                                                             trn_grp_list, trn_grp_img_dir, trn_grp_lbl_dir))
                pool.close()
                pool.join()
                # Loop control
                print("    Augmented dataset No.%0.3d is ready! (%d of %d)" % (n, n, aug_nr))
                n += 1
            print("Defined sets for [%s] created! (%d of %d)" % (trn_dat_name[i], i + 1, len(trn_dat_path)))
        print()

    # Validation set creation
    if vld_set:
        print("Creating defined validation sets:\n----------------------------------------------------------------")
        for i in range(len(vld_dat_path)):
            if i != 0:
                print("- - - - - - - - - - - - - - - -")
            print("Creating validation set for dataset [%s]:" % vld_dat_name[i])
            # Start parallel processing
            pool = mp.Pool(workers)
            for j in vld_n_list[i]:
                pool.apply_async(imhml_seqvld_set, args=(j, i, vld_img_list, vld_lbl_list, vld_dat_path, vld_dat_name,
                                                         vld_grp_list, vld_grp_img_dir, vld_grp_lbl_dir))
            pool.close()
            pool.join()
            print("Defined sets for [%s] created! (%d of %d)" % (vld_dat_name[i], i + 1, len(vld_dat_path)))
        print()

    # Cross-validation set creation
    if tst_set:
        print("Creating defined testing sets:"
              "\n----------------------------------------------------------------")
        for i in range(len(tst_dat_path)):
            if i != 0:
                print("- - - - - - - - - - - - - - - -")
            print("Creating testing set for dataset [%s]:" % tst_dat_name[i])
            # Start parallel processing
            pool = mp.Pool(workers)
            for j in tst_n_list[i]:
                pool.apply_async(imhml_seqtst_set, args=(j, i, tst_img_list, tst_lbl_list, tst_dat_path, tst_dat_name,
                                                         tst_grp_list, tst_grp_img_dir, tst_grp_lbl_dir))
            pool.close()
            pool.join()
            print("Defined sets for [%s] created! (%d of %d)" % (tst_dat_name[i], i + 1, len(tst_dat_path)))
