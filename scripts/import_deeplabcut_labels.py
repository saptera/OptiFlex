import os
import csv
import copy
from OptiFlex.utils.base_func import make_outdirs
from OptiFlex.data.lbl_proc import jsl_write

"""This SCRIPT imports/converts DeepLabCut labels into JSON type of labels.
"""
""" Parameters:
    dlc_lbl_csv: {STR} DeepLabCut label CSV file.
    output_dir: {STR} Output directory for converted data.
"""

# Parameters input  -------------------------------------------------------------------------------------------------- #
dlc_lbl_csv = "./dlc_data.csv"
output_dir = "./imported_labels"
# -------------------------------------------------------------------------------------------------------------------- #

# Create output directory
make_outdirs(output_dir, "Invalid output directory for imported DeepLabCut labels!")

# Converting data
lbl_name = []    # INIT VAR
lbl_temp = {}    # INIT VAR
with open(dlc_lbl_csv, 'r') as lf:
    i = 0
    for row in csv.reader(lf):
        lbl_list = []    # INIT/RESET VAR
        if i == 0:    # First row in CSV is the labeler
            print("DeepLabCut label file created by: %s." % row[1])
        elif i == 1:    # Second row in CSV is the label names
            [lbl_name.append(row[j]) for j in range(len(row)) if j % 2]
        elif i > 2:    # Label data start from forth row
            jsl_name = os.path.splitext(os.path.split(row[0])[1])[0] + '.json'    # First column is file name
            for j in range(1, len(row), 2):
                # Get label name
                lbl_temp['label'] = lbl_name[(j - 1) // 2]
                # Get label X-axis info
                if len(row[j]) == 0:
                    lbl_temp['left'] = None
                    lbl_temp['width'] = 0
                else:
                    lbl_temp['left'] = int(float(row[j]))    # Pixel location is INT
                    lbl_temp['width'] = 1
                    # Get label Y-axis info
                if len(row[j + 1]) == 0:
                    lbl_temp['top'] = None
                    lbl_temp['height'] = 0
                else:
                    lbl_temp['top'] = int(float(row[j + 1]))    # Pixel location is INT
                    lbl_temp['height'] = 1
                # Copy label to the list
                lbl_list.append(copy.deepcopy(lbl_temp))
            jsl_write(os.path.join(output_dir, jsl_name), lbl_list)    # Save to file
        i += 1
