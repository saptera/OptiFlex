import sys
import os
import pathlib
import csv

"""Function list:
x64_sys(): Check if current system architecture is 64-bit based.
mk_outdir(out_dir, err_msg):  Create an output directory for data.
make_outdirs(out_dir, err_msg):  Recursive create an output leaf directory for data.
altmk_outdirs(out_dir, alt_dir, err_msg):  Recursive create an output leaf directory with alternative directory.
read_pathlist_file(list_file):  Read a path list from a CSV file.
prog_print(iteration, total, prefix, suffix):  Create a terminal progress bar for a loop.
os_rand_num(size, norm, digits):  Generate cryptographically secure pseudo-random number.
os_rand_range(lower, higher, size, digits):  Generate cryptographically secure pseudo-random number in a range.
rand_color(mode, norm):  Generate randomized RGB or RGBA color code.
search_files(base_dir, fpre, fsuf):  Find all files meets the search conditions.
"""


def x64_sys():
    """ Check if current system architecture is 64-bit based.
    Args:

    Returns:
        bool: True (if system is x64); False (if system is x32)
    """
    return sys.maxsize > 4294967296    # Max size for 32-bit system: 2**32 = 4294967296


def mk_outdir(out_dir, err_msg='Invalid output directory!'):
    """Create an output directory for data.

    Args:
        out_dir (str): Output directory.
        err_msg (str): Error message when creation error happens.

    Returns:
        str: Created output directory.
    """
    if not os.path.isdir(out_dir):    # Check if folder exists
        try:
            os.mkdir(out_dir)
        except OSError:
            print(err_msg)
            exit(-1)
    return out_dir


def make_outdirs(out_dir, err_msg='Invalid output directory!'):
    """Recursive create an output leaf directory for data.

    Args:
        out_dir (str): Output directory.
        err_msg (str): Error message when creation error happens.

    Returns:
        str: Created output directory.
    """
    if not os.path.isdir(out_dir):    # Check if folder exists
        try:
            os.makedirs(out_dir)
        except OSError:
            print(err_msg)
            exit(-1)
    return out_dir


def altmk_outdirs(out_dir, alt_dir, err_msg='Invalid output directory!'):
    """Recursive create an output leaf directory with alternative directory.

    Args:
        out_dir (str): Output directory.
        alt_dir (str): Alternative directory when [out_dir] is missing.
        err_msg (str): Error message when creation error happens.

    Returns:
        str: Created output directory.
    """
    if (out_dir == str()) or (out_dir is None):
        out_path = alt_dir
        if not os.path.isdir(out_path):    # Check again if folder exists
            os.makedirs(out_path)
    elif not os.path.isdir(out_dir):
        try:
            os.makedirs(out_dir)
        except OSError:
            print(err_msg)
            exit(-1)
    return out_dir


def read_pathlist_file(list_file):
    """ Read a path list from a CSV file.

    Args:
        list_file (str): CSV file path.

    Returns:
        tuple[list[str], list[str]]:
            name_list (list[str]): A unique item name list from the CSV file.
            path_list (list[str]): A path list from the CSV file.
    """
    name_list = []    # INIT VAR
    path_list = []    # INIT VAR
    with open(list_file, 'r') as lf:
        for row in csv.reader(lf):
            name_list.append(row[0])
            p = pathlib.PurePath(row[1]).as_posix()    # Return a string representation of the path with forward slashes
            path_list.append(p)
    return name_list, path_list


def prog_print(iteration, total, prefix=str(), suffix=str()):
    """Create a terminal progress bar for a loop.

    Args:
        iteration (int): Current iteration.
        total (int): Total iterations.
        prefix (str): Prefix string of progress bar (default: str()).
        suffix (str): Suffix string of progress bar (default: str()).

    Returns:
    """
    # Basic settings
    decimals = 2  # Decimals in percent completed
    length = 50  # Character length of bar
    fill = '>'  # Bar fill character
    # Create percentage bar
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / total))
    filled = int(length * iteration // total)
    bar = fill * filled + '-' * (length - filled)
    # Print session
    sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    sys.stdout.flush()
    if iteration == total:  # Print a new line at 100%
        print('')


def os_rand_num(size=8, norm=True, digits=None):
    """ Generate randomized number by system random function.

    Args:
        size (int): Byte size of generated random number (default: 8).
        norm (bool): Normalized mode, random range {0, 1], (default: True).
        digits (int or None): If defined, round random number with defined significant digits,
                              only effective when norm = True(default: None).

    Returns:
        float: Randomized normalized number (norm: True; digits: defined or None).
        int: Randomized integer number based on 2 (norm: False).
    """
    rand_num = int.from_bytes(os.urandom(size), byteorder="big")
    if norm:
        size_base = (1 << (size * 8)) - 1    # Calculate maximum number defined by size (each byte = 8 bits)
        rand_num /= size_base
        if digits is not None:
            rand_num = round(rand_num, digits)
    return rand_num


def os_rand_range(lower, higher, size=8, digits=None):
    """ Generate cryptographically secure pseudo-random number in a range.

    Args:
        lower (float): Lower bound of the required range.
        higher (float): Higher bound of the required range.
        size (int): Byte size of generated random number (default: 8).
        digits (int or None): If defined, round random number with defined significant digits (default: None).

    Returns:
        float: Randomized normalized number.
    """
    length = higher - lower
    rand_num = os_rand_num(size=size, norm=True) * length + lower
    if digits is not None:
        rand_num = round(rand_num, digits)
    return rand_num


def rand_color(mode='RGB', norm=False):
    """ Generate randomized RGB or RGBA color code.

    Args:
        mode (str): Either 'RGB' or 'RGBA', otherwise override with 'RGB' (default: 'RGB').
        norm (bool): Normalized color mode, (default: False).

    Returns:
        tuple: Randomized RGB color code.
    """
    mode = str(mode).upper()    # Unify input
    if norm:    # Normalize color codes to [0, 1], generated by system random functions
        rand_r = os_rand_num(size=1, norm=True, digits=4)
        rand_g = os_rand_num(size=1, norm=True, digits=4)
        rand_b = os_rand_num(size=1, norm=True, digits=4)
        alpha = 1
    else:    # Use [0, 255] color codes generated by system random functions
        rand_r = os_rand_num(size=1, norm=False)
        rand_g = os_rand_num(size=1, norm=False)
        rand_b = os_rand_num(size=1, norm=False)
        alpha = 255
    if mode == 'RGBA':
        color = (rand_r, rand_g, rand_b, alpha)
    else:    # If input is not 'RGB' or 'RGBA', override with 'RGB'
        color = (rand_r, rand_g, rand_b)
    return color


def search_files(base_dir, fpre=str(), fsuf=str()):
    """Find all files meets the search conditions.

    Args:
        base_dir (str): The base folder path to search files.
        fpre (str): Prefix of files to be found, use empty string to find all (default: str()).
        fsuf (str): Suffix of files to be found, use empty string to find all (default: str()).

    Returns:
        tuple[list[list[str]], list[str]]:
            flst (list[list[str]]): A list of lists(leaf-folders) with absolute path of files meets search conditions.
            dlst (list[str]): A list of all leaf folder names contains files found.
            --  [flst] and [dlst] have same length, the order of elements are matched.
    """
    # Get files and their leaf-folder path
    flst = []    # INIT VAR
    dlst = []    # INIT VAR
    for path, _, file in os.walk(base_dir):
        tmp_lst = []    # RESET VAR
        for filename in [f for f in file if f.startswith(fpre) and f.endswith(fsuf)]:
            tmp_lst.append(os.path.join(path, filename))
        if tmp_lst:
            flst.append(tmp_lst)    # Get files
            dlst.append(path)    # Get leaf-folder
    # Trim the common prefix of leaf-folders
    base = os.path.commonpath(dlst)
    for i in range(len(dlst)):
        dlst[i] = os.path.relpath(dlst[i], base)
    return flst, dlst
