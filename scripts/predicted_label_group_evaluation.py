import os
import statistics as stat
from data.lbl_proc import jsl_read
from utils.eval_func import get_ref_lbl, predict_eval
from utils.base_func import prog_print

"""This SCRIPT evaluate the quality of predicted labels.
"""
""" Parameters:
        ref_dir: {STR} Folder with manual labelled reference label files.
        prd_dir: {STR} Folder with predicted label files from model.
        mode: {INT} Evaluation method to be used for qualification:
                --  0 = Distance between PREDICTED and REFERENCE center (ratio)
                --  1 = Distance between PREDICTED and REFERENCE center (absolute)
                --  2 = Area of intersection of PREDICTED and REFERENCE bounding box (ratio)
                --  3 = Area of intersection of PREDICTED and REFERENCE bounding box (absolute)
                --  4 = Mean absolute error (MAE)
"""

# Parameters input  -------------------------------------------------------------------------------------------------- #
ref_dir = './ref_lbl/'
prd_dir = './prd_lbl/'
mode = 4
# -------------------------------------------------------------------------------------------------------------------- #


# Name list for evaluation methods
mode_list = ['Distance Ratio', 'L2 Norm', 'Intersection Ratio', 'Intersection Area', 'MAE']

# Get all required labels.
lbl_list = [f for f in os.listdir(ref_dir) if f.endswith('.json')]
ref_list = [os.path.join(ref_dir, f) for f in lbl_list]
prd_list = [os.path.join(prd_dir, f) for f in lbl_list]

# Main process section
n = len(lbl_list)    # INIT VAR
err_list = []    # INIT VAR
print('Evaluating %d labels.' % n)
for i in range(n):
    if os.path.isfile(prd_list[i]):
        # Read in label files
        ref_lbl = jsl_read(ref_list[i])
        prd_lbl = jsl_read(prd_list[i])
        # Compute evaluation
        ref = get_ref_lbl(ref_lbl)
        qual = predict_eval(prd_lbl, ref, mode)
        # Sum up
        err = sum([qual[e] for e in qual]) / len(qual)
        err_list.append(err)
    else:
        print('Predicted label [%s] is missing!' % lbl_list[i])
    # Progress report
    prog_print(i + 1, n, 'Processed:', '  ')

# Summary and report
mean_err = stat.mean(err_list)
sd_err = stat.stdev(err_list)
print('--------')
print('The MEAN of ERROR in predicted label with [%s] method is: %f.' % (mode_list[mode], mean_err))
print('The STDEV of ERROR in predicted label with [%s] method is: %f.' % (mode_list[mode], sd_err))
