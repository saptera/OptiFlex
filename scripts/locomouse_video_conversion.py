import os
import multiprocessing as mp
import cv2
from utils.base_func import mk_outdir
from data.img_proc import img_bkgsub
from data.vid_proc import get_frm, frm_split
from data.locomouse_func import locomouse_paracsv_read, locomouse_tail_sort

"""This SCRIPT automatically process video files from LocoMouse system and sorting their standard JSON files.
   All folders defined including sub-folders will be scanned!
   Set 'labelled' to [False] to process videos only.
"""
""" Parameters:
      # LocoMouse dataset parameters
        locomouse_vid_path: {STR} Folder with LocoMouse video files with their 'info.csv' file.
        sub_bkg: {BOOL} Defines if a background image exist and will be used to subtract from frames.
      # LocoMouse label file parameters
        labelled: {BOOL} Defines if LocoMouse videos to be processed is manually labelled or not.
        n_tail: {INT} Number of [Tail] label points used in LocoMouse manual labels (effective if [labelled] is True).
      # Parallel processing settings
        workers: {INT} Number of logical cores to be used in pool.
"""

# Parameters input  -------------------------------------------------------------------------------------------------- #
# LocoMouse dataset parameters
locomouse_vid_path = './locomouse/'
sub_bkg = True
# LocoMouse label file parameters
labelled = False
n_tail = 3
# Parallel processing settings
workers = 8
# -------------------------------------------------------------------------------------------------------------------- #


# Define functions for parallel processing
def lm_vid_proc(proc_index, out_dir, vid_file, spt_pos, bkg_img=None):
    # Get frame with defined index
    vid_cap = cv2.VideoCapture(vid_file)
    frm = get_frm(vid_cap, proc_index)
    # Subtract background
    if sub_bkg:
        frm = img_bkgsub(frm, bkg_img)
    # Split frame and save views
    sd_frm, btm_frm = frm_split(frm, spt_pos, 1)
    cv2.imwrite(os.path.join(out_dir, 'sd_frm_{0:05d}.png'.format(proc_index + 1)), sd_frm)
    cv2.imwrite(os.path.join(out_dir, 'btm_frm_{0:05d}.png'.format(proc_index + 1)), btm_frm)


def lm_lbl_proc(proc_index, lbl_list, frm_width):
    locomouse_tail_sort(lbl_list[proc_index], n_tail, frm_width)


# Parallel processing session
if __name__ == '__main__':
    # Get all video files in defined directory
    lm_vids = []  # INIT VAR
    for path, _, file in os.walk(locomouse_vid_path):
        for filename in [f for f in file if f.endswith(".avi")]:
            lm_vids.append(path + '/' + filename)
    n = len(lm_vids)

    # Setting up parallel process
    log_core = mp.cpu_count()
    if workers > log_core:
        print('Not enough logic cores available as defined, using all available cores!')
        workers = log_core
    print('Detected %d logic cores, connecting to %d workers, pool starting...' % (log_core, workers))

    # Major processing loop
    i = 1
    for vid in lm_vids:
        # Get basic info
        print('Processing LocoMouse file: [%s]...' % vid)
        base_dir = os.path.split(vid)[0]
        spt_dir = mk_outdir(base_dir + '/split_view/', err_msg='Invalid split frame directory, function out!')

        # Set capture properties
        cap = cv2.VideoCapture(vid)
        vid_width = int(cap.get(3))    # cv2.VideoCapture::get - propId 3, CV_CAP_PROP_FRAME_WIDTH
        tot_frm = int(cap.get(7))    # cv2.VideoCapture::get - propId 7, CV_CAP_PROP_FRAME_COUNT
        # Read recording information
        info_csv = base_dir + '/info.csv'
        frm_init, frm_step, frm_stop, split_pos = locomouse_paracsv_read(info_csv)
        frm_init -= 1    # Frame count index start @ 1, while OpenCV index start @ 0
        frm_stop = tot_frm if frm_stop is None else frm_stop
        # Get background image
        if sub_bkg:
            bkg_file = base_dir + '/bkg.png'
            bkg = cv2.imread(bkg_file, -1)    # cv::ImreadModes - enum -1, cv2.IMREAD_UNCHANGED
        else:
            bkg = None
        # Parallel process frames
        pool = mp.Pool(workers)
        for idx in range(frm_init, frm_stop, frm_step):
            pool.apply_async(lm_vid_proc, args=(idx, spt_dir, vid, split_pos, bkg))
        pool.close()
        pool.join()

        # Sorting label files for multiple [Tail] tags
        if labelled:
            print('    Sorting and renaming label tags in corresponding label files.')
            lm_lbls = [(spt_dir + f) for f in os.listdir(spt_dir) if f.endswith('.json')]
            # Parallel process labels
            pool = mp.Pool(workers)
            for idx in range(len(lm_lbls)):
                pool.apply_async(lm_lbl_proc, args=(idx, lm_lbls, vid_width))
            pool.close()
            pool.join()

        # Reporting and loop control
        print('    Process fished successfully! (%d of %d)' % (i, n))
        i += 1
