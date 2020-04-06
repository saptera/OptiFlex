import os
import cv2
import numpy as np
from scipy import stats

"""Function list:
get_frm(vid_cap, frm_idx):  Get specified frame of video file.
vid_frm_get(vid_file, out_dir, frm_init, frm_stop, frm_step):  Split video file in to frames as image(*.png).
frm_split(img, split_pos, mode):  Split video frame image files into 2 separated parts.
get_fixcam_bkg(frms, mode):  Compute a possible background image from sequential frames of a fixed camera video file.
img2vid(img_list, vid_name, fps, vid_path):  Create a lossless AVI video file with given images.
img2mp4(img_list, vid_name, fps, vid_path):  Create a MP4 video file with given images.
"""


def get_frm(vid_cap, frm_idx):
    """ Get specified frame of video file.

    Args:
        vid_cap (cv2.VideoCapture): OpenCV video capture ID.
        frm_idx (int): Index of frame to get.

    Returns:
        np.ndarray: {2D, 3D} Image array of specified frame.
    """
    # Verify frame index input
    tot_frm = int(vid_cap.get(7)) - 1    # cv2.VideoCapture::get - propId 7, CV_CAP_PROP_FRAME_COUNT
    if frm_idx < 0:
        print("Frame index must be positive integer, force to 0.")
        frm_idx = 0
    elif frm_idx > tot_frm:
        print("Input frame index larger than total frames, get last frame.")
        frm_idx = tot_frm
    # Get frame
    vid_cap.set(1, frm_idx)    # cv2.VideoCapture::set - propId 1, CV_CAP_PROP_POS_FRAMES
    _, frm = vid_cap.read()
    return frm


def vid_frm_get(vid_file, out_dir=None, frm_init=1, frm_stop=None, frm_step=1):
    """ Split video file in to frames as image(*.png).

    Args:
        vid_file (str): Video file to be split in to frames.
        out_dir (str or None): Output directory to save all acquired frames (default: source folder).
        frm_init (int): The starting index of frame to be acquired (default: 1 [first]).
        frm_stop (int or None): The last index of frame to be acquired (default: None [last]).
        frm_step (int): Frame fixed sampling rate (default: 1 [each]).

    Returns:
        tuple[int, int]:
            frm_width (int): Video file frame width.
            frm_height (int): Video file frame height.
    """
    capture = cv2.VideoCapture(vid_file)
    # Get basic info
    frm_width = int(capture.get(3))    # cv2.VideoCapture::get - propId 3, CV_CAP_PROP_FRAME_WIDTH
    frm_height = int(capture.get(4))    # cv2.VideoCapture::get - propId 4, CV_CAP_PROP_FRAME_HEIGHT
    tot_frm = int(capture.get(7))    # cv2.VideoCapture::get - propId 7, CV_CAP_PROP_FRAME_COUNT

    # Initial frame number [frm_init] verification
    if frm_init < 1:
        print("The starting index of frame should be an integer EQUAL OR LARGER than 1, using 1 instead!")
        frm_init = 0
    else:
        frm_init -= 1    # Frame count index start @ 1, while OpenCV index start @ 0
    # Stop frame number [frm_stop] verification
    if frm_stop is None:
        frm_stop = tot_frm    # [range()] will not include last value
    elif frm_stop > tot_frm - 1:    # OpenCV index start @ 0
        print("Defined ending index of frame was larger than total frames, get until last frame!")
        frm_stop = tot_frm    # [range()] will not include last value
    # Frame step [frm_step] verification
    if frm_step < 1:
        print("The frame sampling step should be an integer EQUAL OR LARGER than 1, using 1 instead!")
        frm_step = 1

    # Frame output loop
    for i in range(frm_init, frm_stop, frm_step):
        capture.set(1, i)    # cv2.VideoCapture::set - propId 1, CV_CAP_PROP_POS_FRAMES
        flag, frm = capture.read()
        if flag:
            img_file = os.path.join(out_dir, "frm_{0:05d}.png".format(i + 1))
            cv2.imwrite(img_file, frm)
        else:
            print("ERROR: [frm_{0:05d}.png] output failed!".format(i + 1))
    return frm_width, frm_height


def frm_split(img, split_pos, mode=1):
    """ Split video frame image files into 2 separated parts by x(width) or y(height) value.

    Args:
        img (np.ndarray): {2D, 3D} Image to be split.
        split_pos (int): The pixel position of the image should be split.
        mode (int): Axis to split image (default: 1).
                --  0 = "x" axis = width
                --  1 = "y" axis = height

    Returns:
        tuple[np.ndarray, np.ndarray]:
            img_uplt (np.ndarray): Upper or left part of splited frame.
            img_lwrt (np.ndarray): Lower or right part of splited frame.
    """
    # Split parameters input verification
    if split_pos <= 0:
        print("Split position must be GREATER than 0, function out!")
        return
    if (mode != 0) and (mode != 1):
        print("Split mode MUST be 0='x' or 1='y', override with default value:1='y'.")
        mode = 1

    # Main process section
    img_h, img_w = img.shape[:2]    # Image.shape {3-TUPLE}: height, width, color_space
    if mode == 0:
        if img_w <= split_pos:
            print("Split position must be SMALLER than image width, function out!")
            return
        else:
            img_uplt = img[0:img_h, 0:split_pos]    # Process left part
            img_lwrt = img[0:img_h, split_pos:img_w]    # Process right part
            return img_uplt, img_lwrt
    elif mode == 1:
        if img_h <= split_pos:
            print("Split position must be SMALLER than image height, function out!")
            return
        else:
            img_uplt = img[0:split_pos, 0:img_w]    # Process upper part
            img_lwrt = img[split_pos:img_h, 0:img_w]    # Process lower part
            return img_uplt, img_lwrt


def get_fixcam_bkg(frms, mode=1):
    """ Compute a possible background image from sequential frames of a fixed camera video file.

    Args:
        frms (list[str]): A list of image files of frames.
        mode (int): The method used to compute background image (default: 1 = Median).
                --  0 = Mean
                --  1 = Median
                --  2 = Mode

    Returns:
        np.ndarray: {2D, 3D} Background image array.
    """
    # Get unified properties from first image
    img = cv2.imread(frms[0], -1)    # cv::ImreadModes - enum -1, cv2.IMREAD_UNCHANGED
    height = img.shape[0]
    width = img.shape[1]
    try:
        channel = img.shape[2]
        flag = False
    except IndexError:
        channel = 1    # Handel single channel images
        flag = True

    # Get all frame image data into a 4D-array
    img_lst = np.zeros(shape=(height, width, channel, len(frms)), dtype=np.uint8)    # INIT VAR
    n = 0    # INIT VAR
    for f in frms:
        # Import images
        img = cv2.imread(f, -1)    # cv::ImreadModes - enum -1, cv2.IMREAD_UNCHANGED
        # Merge values
        if flag:
            img_lst[:, :, 0, n] = img    # Handel single channel images
        else:
            img_lst[:, :, :, n] = img
        n += 1

    # Get background image for sequential frames
    if mode == 0:
        bkg = np.mean(img_lst, axis=3)
        bkg = bkg.astype(np.uint8)    # Convert to OpenCV image array (cv2 img.dtype = uint8)
    elif mode == 1:
        bkg = np.median(img_lst, axis=3)
        bkg = bkg.astype(np.uint8)    # Convert to OpenCV image array (cv2 img.dtype = uint8)
    elif mode == 2:
        bkg = stats.mode(img_lst, axis=3)[0]
        bkg = bkg[:, :, :, 0]    # Remove last dimension
    else:
        bkg = np.zeros(shape=(height, width, channel), dtype=np.uint8)    # Return black image
        print("Defined mode not exist!")

    # Return background image
    if flag:
        bkg = bkg[:, :, 0]    # Handel single channel images
    return bkg


def img2vid(img_list, vid_name, fps=30, vid_path=None):
    """Create a lossless AVI video file with given images.

    Args:
        img_list (list[str]): List of image files to create a video.
        vid_name (str): Output video name, without extension.
        fps (int or float): Output video frame rate (default: 30).
        vid_path (str or None): Output video path, use [img_path] when set to emptystring or None (default: None).

    Returns:
        (bool): True if video successfully created.
    """
    # Get video information
    vid_path = os.path.split(img_list[0])[0] if (vid_path == str()) or (vid_path is None) else vid_path
    vid_file = os.path.join(vid_path, vid_name + ".avi")
    frm_size = cv2.imread(img_list[0]).shape[::-1][1:]

    # Write images to a video
    videoWriter = cv2.VideoWriter(vid_file, cv2.VideoWriter_fourcc(*"HFYU"), fps, frm_size)
    for i in img_list:
        img = cv2.imread(i, -1)   # cv::ImreadModes - enum -1, cv2.IMREAD_UNCHANGED
        videoWriter.write(img)
    return os.path.isfile(vid_file)


def img2mp4(img_list, vid_name, fps=30, vid_path=None):
    """Create a MP4 video file with given images.

    Args:
        img_list (list[str]): List of image files to create a video.
        vid_name (str): Output video name, without extension.
        fps (int or float): Output video frame rate (default: 30).
        vid_path (str or None): Output video path, use [img_path] when set to emptystring or None (default: None).

    Returns:
        (bool): True if video successfully created.
    """
    # Get video information
    vid_path = os.path.split(img_list[0])[0] if (vid_path == str()) or (vid_path is None) else vid_path
    vid_file = os.path.join(vid_path, vid_name + ".mp4")
    frm_size = cv2.imread(img_list[0]).shape[::-1][1:]

    # Write images to a video
    videoWriter = cv2.VideoWriter(vid_file, cv2.VideoWriter_fourcc(*"mp4v"), fps, frm_size)
    for i in img_list:
        img = cv2.imread(i, -1)   # cv::ImreadModes - enum -1, cv2.IMREAD_UNCHANGED
        videoWriter.write(img)
    return os.path.isfile(vid_file)
