import os
import multiprocessing as mp
import numpy as np
import cv2
from OptiFlex.utils.base_func import os_rand_range, make_outdirs, read_pathlist_file
from OptiFlex.data.img_proc import imjsl_resize, imjsl_aug, imhml_resize, imhml_flip, imhml_rotate, imhml_aug
from OptiFlex.data.lbl_proc import jsl_read, jsl_write, jsl_verify, hml_read, hml_write, hml_verify
from OptiFlex.gui import setcrt_gui_glbv

setcrt_gui_glbv.init()
# Parallel processing settings
workers = setcrt_gui_glbv.workers
# Dataset list settings
trn_lst = setcrt_gui_glbv.trn_lst
trn_flg = setcrt_gui_glbv.trn_flg
vld_lst = setcrt_gui_glbv.vld_lst
vld_flg = setcrt_gui_glbv.vld_flg
tst_lst = setcrt_gui_glbv.tst_lst
tst_flg = setcrt_gui_glbv.tst_flg
# Saving settings
out_dir = setcrt_gui_glbv.out_dir
size = setcrt_gui_glbv.size
keep_dum = setcrt_gui_glbv.keep_dum
force_tst = setcrt_gui_glbv.force_tst
# Tagging settings
lbl_std = setcrt_gui_glbv.lbl_std
group_list = setcrt_gui_glbv.group_list
# Label feature settings
js_type = setcrt_gui_glbv.js_type
hm_type = setcrt_gui_glbv.hm_type
lbl_dummy = setcrt_gui_glbv.lbl_dummy
peak = setcrt_gui_glbv.peak
hm_seq = setcrt_gui_glbv.hm_seq
# Augmentation settings
aug_nr = setcrt_gui_glbv.aug_nr
rnd_flp = setcrt_gui_glbv.rnd_flp
axis = setcrt_gui_glbv.axis
rnd_rot = setcrt_gui_glbv.rnd_rot
angle = setcrt_gui_glbv.angle


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
                    if js_type:
                        lbl_list_temp.append(os.path.splitext(f)[0] + ".json")    # Switch extension
                    else:
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


# Define JSON data augmentation function to be parallelled
def imlbl_aug_set(nr, main_index, img_flst, lbl_flst, dir_lst, set_nlst, set_nr, grp_dlst, img_grp_dir, lbl_grp_dir):
    img_file = img_flst[main_index][nr]
    lbl_file = lbl_flst[main_index][nr]
    # Read in files
    img = cv2.imread(os.path.join(dir_lst[main_index], img_file), -1)    # cv::ImreadModes -1, cv2.IMREAD_UNCHANGED
    if os.path.isfile(os.path.join(dir_lst[main_index], lbl_file)):    # Check if some label file is missing
        lbl = jsl_read(os.path.join(dir_lst[main_index], lbl_file))
    else:
        lbl = []
    # Process augmentation
    img, lbl = imjsl_aug(img, lbl, size, axis, angle, rnd_flp, rnd_rot)
    # Standardize label output
    lbl, has_dum = jsl_verify(lbl, lbl_std, lbl_dummy)
    # Output files
    if keep_dum or (not has_dum):    # Converse implication
        # Format output file name
        img_out_file = "%s_%0.3d_%s" % (set_nlst[main_index], set_nr, img_file)
        lbl_out_file = "%s_%0.3d_%s" % (set_nlst[main_index], set_nr, lbl_file)
        # Save by group
        grp_def = grp_dlst[main_index][nr]
        cv2.imwrite(os.path.join(img_grp_dir[grp_def], img_out_file), img)
        jsl_write(os.path.join(lbl_grp_dir[grp_def], lbl_out_file), lbl)


# Define JSON validation set creation function to be parallelled
def imlbl_vld_set(nr, main_index, img_flst, lbl_flst, dir_lst, set_nlst, grp_dlst, img_grp_dir, lbl_grp_dir):
    img_file = img_flst[main_index][nr]
    lbl_file = lbl_flst[main_index][nr]
    # Read in files
    img = cv2.imread(os.path.join(dir_lst[main_index], img_file), -1)    # cv::ImreadModes -1, cv2.IMREAD_UNCHANGED
    if os.path.isfile(os.path.join(dir_lst[main_index], lbl_file)):    # Check if some label file is missing
        lbl = jsl_read(os.path.join(dir_lst[main_index], lbl_file))
    else:
        lbl = []
    # Process resize only to images for validation sets
    img, lbl = imjsl_resize(img, lbl, size, 3)    # interpolation = 3: INTER_AREA
    # Standardize label output
    lbl, has_dum = jsl_verify(lbl, lbl_std, lbl_dummy)
    # Output files
    if keep_dum or (not has_dum):    # Converse implication
        # Format output file name
        img_out_file = "%s_vld_%s" % (set_nlst[main_index], img_file)
        lbl_out_file = "%s_vld_%s" % (set_nlst[main_index], lbl_file)
        # Save by group
        grp_def = grp_dlst[main_index][nr]
        cv2.imwrite(os.path.join(img_grp_dir[grp_def], img_out_file), img)
        jsl_write(os.path.join(lbl_grp_dir[grp_def], lbl_out_file), lbl)


# Define JSON testing set creation function to be parallelled
def imlbl_tst_set(nr, main_index, img_flst, lbl_flst, dir_lst, set_nlst, grp_dlst, img_grp_dir, lbl_grp_dir):
    img_file = img_flst[main_index][nr]
    lbl_file = lbl_flst[main_index][nr]
    # Read in files
    img = cv2.imread(os.path.join(dir_lst[main_index], img_file), -1)    # cv::ImreadModes -1, cv2.IMREAD_UNCHANGED
    if os.path.isfile(os.path.join(dir_lst[main_index], lbl_file)):    # Check if some label file is missing
        lbl = jsl_read(os.path.join(dir_lst[main_index], lbl_file))
    else:
        lbl = []
    # Process resize only to images for testing sets
    img, lbl = imjsl_resize(img, lbl, size, 3)    # interpolation = 3: INTER_AREA
    # Standardize label output
    lbl, has_dum = jsl_verify(lbl, lbl_std, lbl_dummy)
    # Output files
    grp_def = grp_dlst[main_index][nr]
    if force_tst or (keep_dum or (not has_dum)):    # Test set saving logic
        # Format output file name
        img_out_file = "%s_tst_%s" % (set_nlst[main_index], img_file)
        lbl_out_file = "%s_tst_%s" % (set_nlst[main_index], lbl_file)
        # Save by group
        cv2.imwrite(os.path.join(img_grp_dir[grp_def], img_out_file), img)
        jsl_write(os.path.join(lbl_grp_dir[grp_def], lbl_out_file), lbl)


# Define HeatMap data augmentation function to be parallelled
def imhml_aug_set(nr, main_index, img_flst, lbl_flst, dir_lst, set_nlst, set_nr, grp_dlst, img_grp_dir, lbl_grp_dir):
    img_file = img_flst[main_index][nr]
    lbl_file = lbl_flst[main_index][nr]
    # Read in files
    img = cv2.imread(os.path.join(dir_lst[main_index], img_file), -1)    # cv::ImreadModes -1, cv2.IMREAD_UNCHANGED
    if os.path.isfile(os.path.join(dir_lst[main_index], lbl_file)):    # Check if some label file is missing
        lbl = hml_read(os.path.join(dir_lst[main_index], lbl_file))
    else:
        lbl = []
    # Process augmentation
    img, lbl = imhml_aug(img, lbl, size, axis, angle, peak, rnd_flp, rnd_rot)
    # Standardize label output
    lbl, has_dum = hml_verify(lbl, lbl_std, lbl_dummy)
    # Output files
    if keep_dum or (not has_dum):    # Converse implication
        # Format output file name
        img_out_file = "%s_%0.3d_%s" % (set_nlst[main_index], set_nr, img_file)
        lbl_out_file = "%s_%0.3d_%s" % (set_nlst[main_index], set_nr, lbl_file)
        # Save by group
        grp_def = grp_dlst[main_index][nr]
        cv2.imwrite(os.path.join(img_grp_dir[grp_def], img_out_file), img)
        hml_write(os.path.join(lbl_grp_dir[grp_def], lbl_out_file), lbl)


# Define HeatMap sequential data augmentation function to be parallelled
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
    lbl, has_dum = hml_verify(lbl, lbl_std, lbl_dummy)
    # Output files
    if keep_dum or (not has_dum):    # Converse implication
        # Format output file name
        img_out_file = "%s_%0.3d_%s" % (set_nlst[main_index], set_nr, img_file)
        lbl_out_file = "%s_%0.3d_%s" % (set_nlst[main_index], set_nr, lbl_file)
        # Save by group
        grp_def = grp_dlst[main_index][nr]
        cv2.imwrite(os.path.join(img_grp_dir[grp_def], img_out_file), img)
        hml_write(os.path.join(lbl_grp_dir[grp_def], lbl_out_file), lbl)


# Define HeatMap validation set creation function to be parallelled
def imhml_vld_set(nr, main_index, img_flst, lbl_flst, dir_lst, set_nlst, grp_dlst, img_grp_dir, lbl_grp_dir):
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
    lbl, has_dum = hml_verify(lbl, lbl_std, lbl_dummy)
    # Output files
    if keep_dum or (not has_dum):    # Converse implication
        # Format output file name
        img_out_file = "%s_vld_%s" % (set_nlst[main_index], img_file)
        lbl_out_file = "%s_vld_%s" % (set_nlst[main_index], lbl_file)
        # Save by group
        grp_def = grp_dlst[main_index][nr]
        cv2.imwrite(os.path.join(img_grp_dir[grp_def], img_out_file), img)
        hml_write(os.path.join(lbl_grp_dir[grp_def], lbl_out_file), lbl)


# Define HeatMap testing set creation function to be parallelled
def imhml_tst_set(nr, main_index, img_flst, lbl_flst, dir_lst, set_nlst, grp_dlst, img_grp_dir, lbl_grp_dir):
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
    lbl, has_dum = hml_verify(lbl, lbl_std, lbl_dummy)
    # Output files
    grp_def = grp_dlst[main_index][nr]
    if force_tst or (keep_dum or (not has_dum)):    # Test set saving logic
        # Format output file name
        img_out_file = "%s_tst_%s" % (set_nlst[main_index], img_file)
        lbl_out_file = "%s_tst_%s" % (set_nlst[main_index], lbl_file)
        # Save by group
        cv2.imwrite(os.path.join(img_grp_dir[grp_def], img_out_file), img)
        hml_write(os.path.join(lbl_grp_dir[grp_def], lbl_out_file), lbl)


def main():
    global trn_lst, trn_flg, vld_lst, vld_flg, tst_lst, tst_flg, lbl_std, group_list, keep_dum, force_tst, workers, \
        js_type, hm_type, hm_seq, lbl_dummy, aug_nr, rnd_flp, axis, rnd_rot, angle, peak, size, out_dir
    print("Process started!\n")

    # Pre-processing section ---------------------------------------------------------------------------------------
    print("Pre-processing:\n----------------------------------------------------------------")
    # Check training dataset definition
    if trn_flg:
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
    else:
        print("No dataset defined for training!")
        trn_set = False

    # Check validation dataset definition
    if vld_flg:
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
    else:
        print("No dataset defined for validation!")
        vld_set = False

    # Check testing dataset definition
    if tst_flg:
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
    else:
        print("No dataset defined for testing!")
        tst_set = False

    # Final check before processing
    if not (trn_set or vld_set or tst_set):
        print("Not a single dataset defined, system out!")
        exit()
    else:
        print("- - - - - - - - - - - - - - - -")

    # Generate all required random parameters
    if hm_seq:
        ran_data = set()  # INIT VAR, use [set()] to avoid duplicated operation
        while len(ran_data) < aug_nr:
            ran_temp = rand_trans_gen()
            ran_data.add(ran_temp)
        ax = []  # INIT VAR
        ang = []  # INIT VAR
        for rd in ran_data:  # Get required values out from [set]
            ax.append(rd[0])
            ang.append(rd[1])

    # Setting up parallel process ----------------------------------------------------------------------------------
    print("Setup multiprocessing...")
    log_core = mp.cpu_count()
    if workers > log_core:
        print("    Not enough logic cores available as defined, using all available cores!")
        workers = log_core
    print("    Detected %d logic cores, connecting to %d workers.\n" % (log_core, workers))

    # Major processing loops ---------------------------------------------------------------------------------------
    # Augmentation set creation
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
                if js_type:
                    for j in trn_n_list[i]:
                        pool.apply_async(imlbl_aug_set,
                                         args=(j, i, trn_img_list, trn_lbl_list, trn_dat_path, trn_dat_name,
                                               n, trn_grp_list, trn_grp_img_dir, trn_grp_lbl_dir))
                elif hm_type and (not hm_seq):
                    for j in trn_n_list[i]:
                        pool.apply_async(imhml_aug_set,
                                         args=(j, i, trn_img_list, trn_lbl_list, trn_dat_path, trn_dat_name,
                                               n, trn_grp_list, trn_grp_img_dir, trn_grp_lbl_dir))
                else:
                    for j in trn_n_list[i]:
                        pool.apply_async(imhml_seqaug_set,
                                         args=(j, i, trn_img_list, trn_lbl_list, trn_dat_path, trn_dat_name, n,
                                               ax[n - 1], ang[n - 1], trn_grp_list, trn_grp_img_dir, trn_grp_lbl_dir))
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
            if js_type:
                for j in vld_n_list[i]:
                    pool.apply_async(imlbl_vld_set, args=(j, i, vld_img_list, vld_lbl_list, vld_dat_path, vld_dat_name,
                                                          vld_grp_list, vld_grp_img_dir, vld_grp_lbl_dir))
            else:
                for j in vld_n_list[i]:
                    pool.apply_async(imhml_vld_set, args=(j, i, vld_img_list, vld_lbl_list, vld_dat_path, vld_dat_name,
                                                          vld_grp_list, vld_grp_img_dir, vld_grp_lbl_dir))
            pool.close()
            pool.join()
            print("Defined sets for [%s] created! (%d of %d)" % (vld_dat_name[i], i + 1, len(vld_dat_path)))
        print()

    # Testing set creation
    if tst_set:
        print("Creating defined testing sets:"
              "\n----------------------------------------------------------------")
        for i in range(len(tst_dat_path)):
            if i != 0:
                print("- - - - - - - - - - - - - - - -")
            print("Creating testing set for dataset [%s]:" % tst_dat_name[i])
            # Start parallel processing
            pool = mp.Pool(workers)
            if js_type:
                for j in tst_n_list[i]:
                    pool.apply_async(imlbl_tst_set, args=(j, i, tst_img_list, tst_lbl_list, tst_dat_path, tst_dat_name,
                                                          tst_grp_list, tst_grp_img_dir, tst_grp_lbl_dir))
            else:
                for j in tst_n_list[i]:
                    pool.apply_async(imhml_tst_set, args=(j, i, tst_img_list, tst_lbl_list, tst_dat_path, tst_dat_name,
                                                          tst_grp_list, tst_grp_img_dir, tst_grp_lbl_dir))
            pool.close()
            pool.join()
            print("Defined sets for [%s] created! (%d of %d)" % (tst_dat_name[i], i + 1, len(tst_dat_path)))

    # Clean up
    os.remove("gui/setcrt_gui_glbv.py")
