import sys
import os

"""This SCRIPT delete files with defined [prefix] and [suffix], to be used for dataset clean up.
"""
""" Parameters:
        file_path: {STR} Folder with files to be processed.
        prefix: {LIST of STR} Prefix of files to be deleted or kept.
        suffix: {LIST of STR} Suffix of files to be deleted or kept.
        reverse: {BOOL} Defines if defined files will be deleted (FALSE) or kept (TRUE).
"""

# Parameters input  -------------------------------------------------------------------------------------------------- #
file_path = './dataset/'
prefix = ['']
suffix = ['.png']
reverse = False
# -------------------------------------------------------------------------------------------------------------------- #


# Find all files meets the definition
sel = []    # INIT VAR
# Select required files
for pf in prefix:
    for sf in suffix:
        temp = [f for f in os.listdir(file_path) if f.startswith(pf) and f.endswith(sf)]
        sel.extend(temp)
# Change selection behavior depending on [reverse]
if reverse:
    files = os.listdir(file_path)
    for f in sel:
        files.remove(f)
else:
    files = sel

# Executing deletion
n = len(files)
for i in range(n):
    f = os.path.join(file_path, files[i])
    os.remove(f)
    # Report print
    sys.stdout.write('\rDeleting: %d / %d ...' % (i + 1, n))
    sys.stdout.flush()
print('\nProcess DONE!')
