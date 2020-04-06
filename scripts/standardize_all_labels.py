import os
from data.lbl_proc import jsl_read, jsl_write, jsl_verify
from utils.base_func import altmk_outdirs, prog_print

"""This SCRIPT automatically verifies all required labels in defined folder.
   Images and corresponding label files must be in the same folder!
"""
""" Parameters:
        img_path: {STR} Folder with all images.
        lbl_path: {STR} Folder with all corresponding labels (default: img_path).
        out_path: {STR} Folder for outputting verified label files (default: img_path + '/ver_lbl/').
        lbl_list: {LIST of STR} List of all label tags used in labelling, including duplicate, order sensitive.
        lbl_dummy: {DICT} Dictionary used to override when some label is missing.
        keep_dum: {BOOL} Defines if a image/label pair with dummy data inserted will be saved.
"""

# Parameters input  -------------------------------------------------------------------------------------------------- #
img_path = './data/'
lbl_path = ''
out_path = ''
lbl_list = ['Front Right Paw', 'Hind Right Paw', 'Front Left Paw', 'Hind Left Paw', 'Snout',
            'Tail 01', 'Tail 02', 'Tail 03']
lbl_dummy = {'left': -1, 'top': -1, 'width': 0, 'height': 0, 'label': 'NULL'}
keep_dum = True
# -------------------------------------------------------------------------------------------------------------------- #


# Get all image files names in the folder, switch to corresponding label file
files = os.listdir(img_path)
lbl_file = []    # INIT VAR
for file_name in files:
    if file_name.endswith('.png'):
        lbl_file_name = os.path.splitext(file_name)[0] + '.json'    # Switch extension
        lbl_file.append(lbl_file_name)

# Check label directory
if (lbl_path == str()) or (lbl_path is None):
    lbl_path = img_path
elif not os.path.isdir(lbl_path):
    print('Defined label directory NOT EXIST!')
    exit()
if not lbl_path.endswith('/'):
    lbl_path += '/'

# Set output directory
out_path = altmk_outdirs(out_path, os.path.join(img_path + 'ver_lbl/'), err_msg='Invalid output path, function out!')
if not out_path.endswith('/'):
    out_path += '/'

# Main process section
n = len(lbl_file)
print('Verifying [%d] label files:' % n)
i = 1    # INIT VAR
for lbl_name in lbl_file:
    if os.path.isfile(lbl_path + lbl_name):
        lbl_data = jsl_read(lbl_path + lbl_name)
    else:
        lbl_data = []
    lbl_out, has_dum = jsl_verify(jsl_data=lbl_data, lbl_list=lbl_list, js_dummy=lbl_dummy)
    if keep_dum or (not has_dum):    # Converse implication
        jsl_write(out_path + lbl_name, lbl_out)
    # Reporting session
    prog_print(i, n, 'Progress:', 'label files processed.')
    i += 1
print('Label file verification DONE!')
