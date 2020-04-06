import os
import multiprocessing as mp
import queue
import cv2
from utils.base_func import read_pathlist_file, make_outdirs, prog_print
from data.lbl_proc import jsl_read, lblconv_json2heat, hml_write, hml_verify

"""This SCRIPT converts JSON label files in a dataset to HeatMap label files.
   All folders defined including sub-folders will be scanned!
"""
""" Parameters:
      # Basic settings
        data_set_csv: {STR} A CSV file defining dataset for label conversion.
        output_dir: {STR} Folder where all converted labels will be stored (default: str()/None [original folder]).
      # Label conversion settings
        zero_offset: {BOOL} Use ["top"] and ["left"] value from JSON labels directly as center of HeatMap when [True].
        bb_meas: {DICT of (TUPLE of 2-INT)} Allowed bounding box size for each type of labels.
        peak {INT or FLOAT}: Peak value of converted HeatMap.
      # Label verification settings
        verify: {BOOL} Defines if converted HeatMap labels should be verified or not.
        lbl_list: {LIST of STR} List of all label tags used in labelling, including duplicate, order sensitive.
        hml_dummy: {2D-ARRAY} HeatMap used to override when some label is missing.
      # Parallel processing settings
        workers: {INT} Number of logical cores to be used in pool.
"""

# Parameters input  -------------------------------------------------------------------------------------------------- #
# Basic settings
data_set_csv = "./dat_lst.csv"
output_dir = ""
# Label conversion settings
zero_offset = True
bb_meas = {}
peak = 16.0
# Label verification settings
verify = False
lbl_std = []
hml_dummy = None
# Parallel processing settings
workers = 8
# -------------------------------------------------------------------------------------------------------------------- #


# Define function for multiprocessing
def lbl_conv(proc_index, major_index, img_flst, jsl_flst, hml_flst):
    # Read in files
    img = cv2.imread(img_flst[major_index][proc_index])
    if os.path.isfile(jsl_flst[major_index][proc_index]):    # Check if some label file is missing
        lbl = jsl_read(jsl_flst[major_index][proc_index])
        if zero_offset:    # Use ["top"] and ["left"] value directly as center of HeatMap
            for lbl_data in lbl:
                lbl_data["width"] = lbl_data["height"] = 0
    else:
        lbl = []
    # Conversion and output
    hml_data = lblconv_json2heat(lbl, img, bb_meas, peak)
    if verify:
        hml_data, _ = hml_verify(hml_data, lbl_std, hml_dummy)
        hml_write(hml_flst[major_index][proc_index], hml_data)
    else:
        if hml_data:
            hml_write(hml_flst[major_index][proc_index], hml_data)
        else:
            pass
    return proc_index


# Parallel processing session
if __name__ == "__main__":
    # Get all path info
    file_path_name, file_path = read_pathlist_file(data_set_csv)    # Get folder and their names

    # Set output folder the as input if [output_dir] is not defined
    if (output_dir == str()) or (output_dir is None):
        output_path = file_path
    # Creat output folders if [output_dir] is defined
    else:
        output_path = [os.path.join(output_dir, p) for p in file_path_name]    # Output leaf directory
        for p in output_path:
            make_outdirs(p, err_msg="Invalid output folder, function out!")

    # Get all required files in the folders
    img_list = []    # INIT VAR
    jsl_list = []    # INIT VAR
    hml_list = []    # INIT VAR
    for i in range(len(file_path)):
        files = os.listdir(file_path[i])    # Get files in sub-folder
        img_list_temp = []    # INIT/RESET VAR
        jsl_list_temp = []    # INIT/RESET VAR
        hml_list_temp = []    # INIT/RESET VAR
        for f in files:
            if f.endswith(".png"):
                img_file_name = os.path.join(file_path[i], f)
                img_list_temp.append(img_file_name)
                # Switch extension to get JSON label files
                jsl_file_name = os.path.splitext(img_file_name)[0] + ".json"
                jsl_list_temp.append(jsl_file_name)
                # Set output HeatMap label files
                hml_file_name = os.path.join(output_path[i], os.path.splitext(f)[0] + ".pkl")
                hml_list_temp.append(hml_file_name)
        # Passing temp list data to the main list
        img_list.append(img_list_temp)
        jsl_list.append(jsl_list_temp)
        hml_list.append(hml_list_temp)

    # Setting up parallel process
    log_core = mp.cpu_count()
    if workers > log_core:
        print("Not enough logic cores available as defined, using all available cores!")
        workers = log_core
    print("Detected %d logic cores, connecting to %d workers, pool starting...\n" % (log_core, workers))
    pool = mp.Pool(workers)
    get_index = queue.Queue()

    # Process conversion
    tot = len(file_path)
    for i in range(tot):
        print("Converting label files of dataset [%s]... (%d / %d)" % (file_path_name[i], i + 1, tot))
        pool = mp.Pool(workers)
        get_index = queue.Queue()
        count = len(img_list[i])
        for j in range(count):
            get_index.put(pool.apply_async(lbl_conv, args=(j, i, img_list, jsl_list, hml_list)))
        pool.close()
        while True:
            n = get_index.get().get() + 1
            prog_print(n, count, "  Progress:", "converted.")
            if n == count:
                break
        pool.join()
    print("Process DONE!")
