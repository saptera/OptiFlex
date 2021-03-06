import os
import multiprocessing as mp
import queue
import cv2
from OptiFlex.utils.base_func import make_outdirs, prog_print
from OptiFlex.data.img_proc import img_bkgsub
from OptiFlex.data.vid_proc import get_fixcam_bkg

"""This SCRIPT preprocess images to meet the requirement of model inference predictions.
   All folders defined including sub-folders will be scanned!
"""
""" Parameters:
      # Basic settings
        data_set_dir: {STR} Folder with sub-folders where all frames are located.
        output_dir: {STR} Folder where all preprocessed frames will be stored (default: str()/None = [original]).
      # Resize setting
        resize: {BOOL} If set to [True], all frames will be resize to the size defined by [size].
        size: {TUPLE of 2-INT} Defines final size of image (in pixel): size[0] = width, size[1] = height.
      # Background subtraction parameters
        group_list: {LIST of STR} Tag names used to separate images into different groups.
        assume_background: {BOOL} If set to [True], try generating background image from frames when its not found.
      # Parallel processing settings
        workers: {INT} Number of logical cores to be used in pool.
"""

# Parameters input  -------------------------------------------------------------------------------------------------- #
# Basic settings
data_set_dir = './data_set/'
output_dir = ''
# Resize setting
resize = True
size = (700, 100)
# Background subtraction parameters
group_list = ['btm', 'sd']
assume_background = True
# Parallel processing settings
workers = 8
# -------------------------------------------------------------------------------------------------------------------- #


# Define function for multiprocessing
def img_preproc(proc_index, major_index, frm_flst, bkg_flst, tag_lst, out_flst):
    # Get current group tag
    curr_tag = tag_lst[major_index][proc_index]
    # Read in files
    bkg = cv2.imread(bkg_flst[curr_tag][major_index])
    img = cv2.imread(frm_flst[major_index][proc_index])
    # Process
    dst = img_bkgsub(img, bkg)
    if resize:
        dst = cv2.resize(src=dst, dsize=size, interpolation=3)    # cv2.resize: interpolation 3 = INTER_AREA
    cv2.imwrite(out_flst[major_index][proc_index], dst)
    return proc_index


# Parallel processing session
if __name__ == '__main__':
    # Get all path info
    file_path = [os.path.join(p, 'split_view/') for p in os.scandir(data_set_dir) if p.is_dir()]    # Get sub-folders
    file_path_name = [os.path.split(p[:-12])[1] for p in file_path]    # Get sub-folder names

    # Get all required background images
    bkg_list = dict()    # INIT VAR
    for tag in group_list:
        bkg_list[tag] = [os.path.join(p, 'bkg_' + tag + '.png') for p in os.scandir(data_set_dir) if p.is_dir()]
        for i in range(len(bkg_list[tag])):
            if not os.path.isfile(bkg_list[tag][i]):
                if assume_background:
                    print('Background [%s] for dataset [%s] missing, try generating!' % (tag, file_path_name[i]))
                    grp_frm = [os.path.join(file_path[i], f) for f in os.listdir(file_path[i])
                               if f.startswith(tag) and f.endswith('.png')]
                    bkg_img = get_fixcam_bkg(grp_frm, 1)    # 1 = Median mode
                    cv2.imwrite(bkg_list[tag][i], bkg_img)
                else:
                    print('Background [%s] for dataset [%s] missing, function out!' % (tag, file_path_name[i]))
                    exit()

    # Set output folder the as input if [output_dir] is not defined
    output_dir = data_set_dir if (output_dir == str()) or (output_dir is None) else output_dir
    # Creat output folders
    output_path = []    # INIT VAR
    for tag in group_list:
        output_path_temp = [os.path.join(output_dir, p, tag) for p in file_path_name]    # Grouped output leaf directory
        output_path.append(output_path_temp)
    for gp in output_path:
        for p in gp:
            make_outdirs(p, err_msg='Invalid output folder, function out!')

    # Get all required frames in the folders
    frm_list = []    # INIT VAR
    imout_list = []    # INIT VAR
    group_tag = []    # INIT VAR
    for i in range(len(file_path)):
        files = os.listdir(file_path[i])    # Get files in sub-folder
        frm_list_temp = []    # INIT/RESET VAR
        imout_list_temp = []    # INIT/RESET VAR
        group_tag_temp = []    # INIT/RESET VAR
        for f in files:
            if f.endswith('.png'):
                frm_list_temp.append(os.path.join(file_path[i], f))
                # Define groups and set out file
                for t in range(len(group_list)):
                    if f.startswith(group_list[t]):
                        group_tag_temp.append(group_list[t])
                        imout_list_temp.append(os.path.join(output_path[t][i], f))
        # Passing temp list data to the main list
        frm_list.append(frm_list_temp)
        imout_list.append(imout_list_temp)
        group_tag.append(group_tag_temp)

    # Setting up parallel process
    log_core = mp.cpu_count()
    if workers > log_core:
        print('Not enough logic cores available as defined, using all available cores!')
        workers = log_core
    print('Detected %d logic cores, connecting to %d workers, pool starting...' % (log_core, workers))

    # Process background subtraction
    tot = len(file_path)
    for i in range(tot):
        pool = mp.Pool(workers)
        get_index = queue.Queue()
        print('Subtracting frame background of dataset [%s]... (%d / %d)' % (file_path_name[i], i + 1, tot))
        count = len(frm_list[i])
        for j in range(count):
            get_index.put(pool.apply_async(img_preproc, args=(j, i, frm_list, bkg_list, group_tag, imout_list)))
        pool.close()
        while True:
            n = get_index.get().get() + 1
            prog_print(n, count, '  Progress:', 'processed.')
            if n == count:
                break
        pool.join()
    print('Process DONE!')
