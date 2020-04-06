import os
from OptiFlex.utils.base_func import make_outdirs, prog_print
from OptiFlex.data.lbl_proc import hml_read, jsl_write, lblconv_heat2json
from OptiFlex.data.post_pred import btm_mirror_paw_fix

"""This SCRIPT automatically correct Bottom-Mirror type setup paw predictions.
"""
""" Parameters:
        btm_pred_dir: {STR} Folder with predicted bottom-view HeatMap label files.
        sd_pred_dir: {STR} Folder with predicted side-view HeatMap label files.
        out_dir: {STR} Output folder for corrected JSON label files.
"""

# Parameters input  -------------------------------------------------------------------------------------------------- #
btm_pred_dir = './pred_btm_hm_lbl/'
sd_pred_dir = './pred_sd_hm_lbl/'
out_dir = './corrected/'
# -------------------------------------------------------------------------------------------------------------------- #


# Create output leaf directories
sd_out_dir = make_outdirs(os.path.join(out_dir, 'sd'), 'Invalid side-view label output directory!')
btm_out_dir = make_outdirs(os.path.join(out_dir, 'btm'), 'Invalid bottom-view label output directory!')
# Get all file paths
sd_pred_lst = []    # INIT VAR
sd_out_lst = []    # INIT VAR
btm_pred_lst = []    # INIT VAR
btm_out_lst = []    # INIT VAR
for f in os.listdir(sd_pred_dir):
    if f.endswith(".pkl"):
        sd_pred_lst.append(os.path.join(sd_pred_dir, f))
        # Switch extension and directory for SD output
        sd_out_lst.append(os.path.join(sd_out_dir, os.path.splitext(f)[0] + ".json"))
        # Replace SD with BTM for bottom-view reference labels
        btm_pred_lst.append(os.path.join(btm_pred_dir, f.replace("_sd_", "_btm_")))
        # Switch extension and directory for BTM output
        btm_out_lst.append(os.path.join(btm_out_dir, os.path.splitext(f.replace("_sd_", "_btm_"))[0] + ".json"))

# Main process loop
n = len(sd_pred_lst)
for i in range(n):
    # Read-in predicted side-view HeatMap labels
    sd_pred_hm = hml_read(sd_pred_lst[i])
    # Compute correction
    if os.path.isfile(btm_pred_lst[i]):
        btm_pred_hm = hml_read(btm_pred_lst[i])
        btm_lbl, sd_lbl = btm_mirror_paw_fix(btm_pred_hm, sd_pred_hm)
    else:
        sd_lbl = lblconv_heat2json(sd_pred_hm, 0.9)
        btm_lbl = []
    # Write output file
    jsl_write(btm_out_lst[i], btm_lbl)
    jsl_write(sd_out_lst[i], sd_lbl)
    # Print progress
    prog_print(i + 1, n, 'Progress:')
