import os
from utils.base_func import make_outdirs
from training.inference import cpm_inference

"""This SCRIPT subtract background from a fixed camera video frames.
   All folders defined including sub-folders will be scanned!
"""
""" Parameters:
      # Basic settings
        target_dir: {STR} Folder with sub-folders where all frames tobe predicted are located.
        output_dir: {STR} Folder where all prediction results will be stored (default: str()/None = [original]).
      # Inference settings
        group_list: {LIST of STR} Tags define images into different groups, different group use different model.
        model_list: {LIST of STR} Models used to process different groups, MUST be same size as [group_list].
        save_json: {BOOL} Will convert HeatMap labels to JSON labels if set [True] (default: True).
        save_image: {BOOL} Will create labelled images if set [True] (default: True).
"""

# Parameters input  -------------------------------------------------------------------------------------------------- #
# Basic settings
target_dir = './tgt/'
output_dir = ''
# Inference settings
group_list = ['btm', 'sd']
model_list = ['btm.model', 'sd.model']
save_json = True
save_image = True
# -------------------------------------------------------------------------------------------------------------------- #


# Check inputs
if len(group_list) != len(model_list):
    print("Parameter [group_list] and [model_list] MUST have SAME length!")
    exit(-1)

# Get all path info
tgt_path = []    # INIT VAR
for tag in group_list:
    path_temp = [os.path.join(p, tag) for p in os.scandir(target_dir) if p.is_dir()]    # Get sub-folders
    tgt_path.append(path_temp)
set_name = [os.path.split(p)[1] for p in os.scandir(target_dir) if p.is_dir()]    # Get dataset names

# Set main output path
if (output_dir is None) or (output_dir == str()):
    base = os.path.split(target_dir)[0] if target_dir.endswith('/') or target_dir.endswith('\\') else target_dir
    output_dir = os.path.join(os.path.split(base)[0], 'predictions')
# Set output leaf directories for all sets
out_path = []    # INIT VAR
for tag in group_list:
    out_path_temp = []    # INIT/RESET VAR
    for n in set_name:
        out_temp = make_outdirs(os.path.join(output_dir, n, tag),
                                err_msg=("Invalid output folder: set [%s], group [%s]." % (n, tag)))
        out_path_temp.append(out_temp)
    out_path.append(out_path_temp)

# Execute predictions
gl = len(group_list)
sl = len(set_name)
tot = gl * sl
for i in range(gl):
    for j in range(sl):
        cpm_inference(model_list[i], tgt_path[i][j], out_path[i][j], save_json, save_image)
        pctg = (i + 1) * (j + 1) / tot * 100
        print("-> Progress: within %d of %d groups, %d of %d sets processed. [%.2f%%]\n" % (i + 1, gl, j + 1, sl, pctg))
print("--------\nAll process DONE!")
