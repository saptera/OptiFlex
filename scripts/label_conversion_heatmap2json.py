import os
import multiprocessing as mp
import queue
from OptiFlex.utils.base_func import read_pathlist_file, make_outdirs, prog_print
from OptiFlex.data.lbl_proc import hml_read, lblconv_heat2json, jsl_write, jsl_verify

"""This SCRIPT converts HeatMap label files in a dataset to JSON label files.
   All folders defined including sub-folders will be scanned!
"""
""" Parameters:
      # Basic settings
        data_set_csv: {STR} A CSV file defining dataset for label conversion.
        output_dir: {STR} Folder where all converted labels will be stored (default: str()/None [original folder]).
      # Label conversion settings
        threshold: {INT or FLOAT} HeatMap peak value threshold.
      # Label verification settings
        verify: {BOOL} Defines if converted HeatMap labels should be verified or not.
        lbl_list: {LIST of STR} List of all label tags used in labelling, including duplicate, order sensitive.
        jsl_dummy: {DICT} Dictionary used to override when some label is missing,
      # Parallel processing settings
        workers: {INT} Number of logical cores to be used in pool.
"""

# Parameters input  -------------------------------------------------------------------------------------------------- #
# Basic settings
data_set_csv = "./dat_lst.csv"
output_dir = ""
# Label conversion settings
threshold = 0.5
# Label verification settings
verify = False
lbl_std = []
jsl_dummy = {"left": None, "top": None, "width": 0, "height": 0, "label": None}
# Parallel processing settings
workers = 8
# -------------------------------------------------------------------------------------------------------------------- #


# Define function for multiprocessing
def lbl_conv(proc_index, major_index, hml_flst, jsl_flst):
    # Read in files
    if os.path.isfile(hml_flst[major_index][proc_index]):    # Check if some label file is missing
        lbl = hml_read(hml_flst[major_index][proc_index])
    else:
        lbl = []
    # Conversion and output
    jsl_data = lblconv_heat2json(lbl, threshold)
    if verify:
        jsl_data, _ = jsl_verify(jsl_data, lbl_std, jsl_dummy)
        jsl_write(jsl_flst[major_index][proc_index], jsl_data)
    else:
        if jsl_data:
            jsl_write(jsl_flst[major_index][proc_index], jsl_data)
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
    hml_list = []    # INIT VAR
    jsl_list = []    # INIT VAR
    for i in range(len(file_path)):
        files = os.listdir(file_path[i])    # Get files in sub-folder
        hml_list_temp = []    # INIT/RESET VAR
        jsl_list_temp = []    # INIT/RESET VAR
        for f in files:
            if f.endswith(".pkl"):
                hml_file_name = os.path.join(file_path[i], f)
                hml_list_temp.append(hml_file_name)
                # Set output JSON label files
                jsl_file_name = os.path.join(output_path[i], os.path.splitext(f)[0] + ".json")
                jsl_list_temp.append(jsl_file_name)
        # Passing temp list data to the main list
        hml_list.append(hml_list_temp)
        jsl_list.append(jsl_list_temp)

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
        count = len(hml_list[i])
        for j in range(count):
            get_index.put(pool.apply_async(lbl_conv, args=(j, i, hml_list, jsl_list)))
        pool.close()
        while True:
            n = get_index.get().get() + 1
            prog_print(n, count, "  Progress:", "converted.")
            if n == count:
                break
        pool.join()
    print("Process DONE!")
