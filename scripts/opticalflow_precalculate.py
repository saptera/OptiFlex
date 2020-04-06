import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import multiprocessing as mp
import zlib
import pickle as pkl
import numpy as np
import cv2
from data.lbl_proc import hml_read, hm_optical_warp
from utils.base_func import make_outdirs

"""This SCRIPT preprocess OpticalFlow for accelerate model training.
"""
""" Parameters:
      # File directory parameters
        img_path: {STR} Folder with image files.
        hml_path: {STR} Folder with corresponding HeatMap label files.
        out_path: {STR} Folder for output OpticalFlow tensor.
      # OpticalFlow parameters
        prefix_length: {INT} Define the length of prefix for identifying different file groups.
        frame_range: {INT} Define total number of neighbouring frames to be calculated with OpticalFlow.
        skip_ratio: {INT} Define the ratio of frames to be skipped.
      # Parallel processing settings
        workers: {INT} Number of logical cores to be used in pool.
"""

# Parameters input  -------------------------------------------------------------------------------------------------- #
img_path = "./img/"
hml_path = "./lbl/"
out_path = "./out/"
prefix_length = 17
frame_range = 4
skip_ratio = 1
workers = 8
# -------------------------------------------------------------------------------------------------------------------- #


def get_proc_lst(img_dir, lbl_dir, out_dir, len_pref, rng=4, skp=1):
    # Get grouped file lists
    dat_sets = sorted(set([f[:len_pref] for f in os.listdir(img_dir) if f.endswith(".png")]))
    img_file = [[] for _ in range(len(dat_sets))]  # INIT VAR
    lbl_file = [[] for _ in range(len(dat_sets))]    # INIT VAR
    for f in os.listdir(img_dir):
        if f.endswith(".png"):
            idx = dat_sets.index(f[:len_pref])
            img_file[idx].append(os.path.join(img_dir, f))
            lbl_file[idx].append(os.path.join(lbl_dir, os.path.splitext(f)[0] + ".pkl"))
    # Get process list
    idx_rng = [i * skp for i in range(-rng, rng + 1)]
    proc_lst = []    # INIT VAR
    for n in range(len(img_file)):
        img_grp = sorted(img_file[n])    # Sort the file list to verify sequence
        lbl_grp = sorted(lbl_file[n])    # Sort the file list to verify sequence
        max_idx = len(img_grp) - 1
        for i in range(max_idx + 1):
            out_file = os.path.join(out_dir, os.path.split(lbl_grp[i])[1])
            of_img_lst = []    # INIT/RESET VAR
            of_lbl_lst = []    # INIT/RESET VAR
            for j in idx_rng:
                if j == 0:    # Padding self
                    of_img_lst.append(None)
                    of_lbl_lst.append(None)
                else:
                    idx = i + j
                    idx = 0 if idx < 0 else idx    # Verify minimum index
                    idx = max_idx if idx > max_idx else idx    # Verify maximum index
                    of_img_lst.append(img_grp[idx])
                    of_lbl_lst.append(lbl_grp[idx])
            proc_lst.append((img_grp[i], lbl_grp[i], of_img_lst, of_lbl_lst, out_file))
    return proc_lst


def hml_warp(img_ref, img_tgt, hml_data):
    # Convert HeatMap label data
    hm_stack = []
    for hm in hml_data:
        hm_stack.append(hm["heatmap"])
    # Perform OpticalFlow
    hm_tensor = np.stack(hm_stack, axis=-1)[:, :, :, None]
    hm_warp = hm_optical_warp(img_ref, img_tgt, hm_tensor)
    return hm_warp


def proc_optical_flow_tensor(proc_unit):
    img_tgt = cv2.imread(proc_unit[0])
    hml_tgt = hml_read(proc_unit[1])
    of_tensor = []
    for i in range(len(proc_unit[2])):
        if proc_unit[2][i] is not None:
            img_ref = cv2.imread(proc_unit[2][i])
            hml_data = hml_read(proc_unit[3][i])
            warp_tensor = hml_warp(img_ref, img_tgt, hml_data)
        else:
            hm_stack = []
            for hm in hml_tgt:
                hm_stack.append(hm["heatmap"])
            warp_tensor = np.stack(hm_stack, axis=-1)
        of_tensor.append(warp_tensor)
    return np.asarray(of_tensor)


def of_tensor_write(of_pkl_file, of_tensor):
    comp = zlib.compress(pkl.dumps(of_tensor, protocol=2))
    with open(of_pkl_file, 'wb') as outfile:
        pkl.dump(comp, outfile, protocol=2)


def mp_of_proc(idx, proc_lst):
    print("Current process #: %05d" % idx)
    of_tensor = proc_optical_flow_tensor(proc_lst[idx])
    of_tensor_write(proc_lst[idx][4], of_tensor)


if __name__ == "__main__":
    make_outdirs(out_path)
    full_list = get_proc_lst(img_path, hml_path, out_path, prefix_length, frame_range, skip_ratio)

    # Setting up parallel process
    log_core = mp.cpu_count()
    if workers > log_core:
        print("Not enough logic cores available as defined, using all available cores!")
        workers = log_core
    print("Detected %d logic cores, connecting to %d workers, pool starting...\n" % (log_core, workers))
    pool = mp.Pool(workers)

    # Processing
    for proc_idx in range(len(full_list)):
        pool.apply_async(mp_of_proc, args=(proc_idx, full_list))
    pool.close()
    pool.join()
