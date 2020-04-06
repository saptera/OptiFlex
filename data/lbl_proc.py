import json
import pickle as pkl
import copy
import zlib
import numpy as np
from scipy import stats
import cv2
from utils.base_func import rand_color

"""Function list:
# JSON type label file functions:
  # IO functions:
    jsl_read(js_lbl_file):  Import a standard JSON label file to a Python list of dictionary.
    jsl_write(dst_jsl_file, jsl_data):  Write JSON label data to a standard JSON label file.
  # Label operation:
    jsl_verify(jsl_data, lbl_list, js_dummy): Verify JSON labels, padding missing labels, standardize label order.
    jsl_resize(jsl_data, src_imsize, dst_imsize, bb_meas):  Resize JSON labels to their corresponding image size.
    jsl_plot(jsl_data, img, color_list, annotate):  Plot JSON label information to corresponding image.
# Label converison functions:
    lblconv_json2heat(jsl_data, img, bb_meas. peak):  Convert JSON type label to HeatMap type label.
    lblconv_heat2json(hml_data, th):  Convert HeatMap type label to JSON type label.
# HeatMap type label file functions:
  # IO functions:
    hml_read(hm_lbl_file):  Import a PICKLE HeatMap label file to a Python list of dictionary.
    hml_write(dst_hml_file, hml_data):  Write HeatMap label data to a PICKLE label file.
  # Label operation:
    hml_verify(hml_data, lbl_list, hm_dummy): Verify HeatMap type labels.
    hml_plot(hml_data, img, color_list): Plot HeatMap label information to corresponding image.
    hml_jntplot(hml_data, img, joint, color_list):  Plot defined HeatMap joint names to corresponding image.
  # OpticalFlow operations:
    hm_optical_warp(img_prev, img_next, hm_tensor):  Compute HeatMap shift between two images using optical-flow.
    hm_multi_warp(img_lst, hm_tensor):  Compute HeatMap shift between a sequence of images using optical-flow.
# Label batch process functions:
    lbl_merge(lbl_file, lbl_list):  Read and merge multiple JSON label files into 2 NumPy nD-arrays.
    lbl_split(lbl_crv_x, lbl_crv_y, lbl_file, lbl_list, bb_meas):  Split and write 2 NumPy nD-arrays back to JSON file.
"""


# JSON type label file functions ------------------------------------------------------------------------------------- #

def jsl_read(js_lbl_file):
    """ Import a standard JSON label file to a Python list of dictionary.

    Args:
        js_lbl_file (str): Labelling file contained with labels.

    Returns:
        list[dict]: List of dictionary with label info.
    """
    with open(js_lbl_file) as infile:
        jsl_data = json.load(infile)
    return jsl_data


def jsl_write(dst_jsl_file, jsl_data):
    """ Write JSON label data to a standard JSON label file.

    Args:
        dst_jsl_file (str): Labelling file to be write label data (*.json).
        jsl_data (list[dict]): List of dictionary with label info.

    Returns:
        bool: File creation status.
    """
    if len(jsl_data) != 0:
        with open(dst_jsl_file, 'w') as outfile:
            json.dump(jsl_data, outfile)
        return True
    else:
        print('Empty label data input, file not created!')
        return False


def jsl_verify(jsl_data, lbl_list, js_dummy=None):
    """ Verify JSON labels, padding missing labels, standardize label order.

    Args:
        jsl_data (list[dict]): List of dictionary with label info.
        lbl_list (list[str]): List of all label tags used in labelling, including duplicates, order sensitive.
        js_dummy (dict or None): Dictionary used to override when some label is missing,
                                 (default: None = {'left': None, 'top': None, 'width': 0, 'height': 0, 'label': None}).

    Returns:
        tuple[list[dict], bool]:
            jsl_out (list[dict]): List of dictionary with verified and standardized label info.
            has_dum (bool): A flag to identify whether [jsl_out] has dummy data insertion or not.
    """
    lbl_copy = copy.deepcopy(jsl_data)    # Make copy of original data, avoid unexpected modification
    if js_dummy is None:    # Use default dummy if not defined
        js_dummy = {'left': None, 'top': None, 'width': 0, 'height': 0, 'label': None}
    lbl_temp = {'left': None, 'top': None, 'width': None, 'height': None, 'label': None}    # INIT VAR
    jsl_out = []    # INIT VAR
    has_dum = False    # INIT VAR, False by default, no dummy data inserted.
    for lbl_name in lbl_list:    # Checking labels in defined sequence
        flag = True    # RESET VAR: (label missing) flag to True (not found yet)
        for lbl in lbl_copy:    # Label searching
            if lbl['label'] == lbl_name:
                lbl_temp = copy.deepcopy(lbl)
                lbl_copy.remove(lbl)    # Avoid getting same label multiple times
                flag = False    # Label found, (label missing) flag set to False
                break    # Avoid pass-over when found
        if flag:    # Padding missing label with dummy
            js_dummy['label'] = lbl_name
            lbl_temp = copy.deepcopy(js_dummy)
            has_dum = True    # Set status to True as dummy data inserted for missing data.
        jsl_out.append(copy.deepcopy(lbl_temp))    # Putting labels in defined order
    return jsl_out, has_dum


def jsl_resize(jsl_data, src_imsize, dst_imsize, bb_meas=None):
    """ Resize JSON labels to their corresponding image size.

    Args:
        jsl_data (list[dict]): List of dictionary with label info.
        src_imsize (tuple[int, int]): Tuple of [0]-width, [1]-height of source image size used in prediction.
        dst_imsize (tuple[int, int]): Tuple of [0]-width, [1]-height of target image size of original.
        bb_meas (dict[str, tuple[int, int]] or None): Define bounding box size for each type of labels (default: None).
                --  KEY (str) = label name
                --  VALUE (tuple[int, int]) = (width, height)

    Returns:
        list[dict]: List of dictionary with resized label info.
    """
    # Get resize matrix info
    resize_mat = np.array([dst_imsize[0]/src_imsize[0], dst_imsize[1]/src_imsize[1]])
    jsl_out = copy.deepcopy(jsl_data)    # INIT VAR
    # Execute label resize -- [x', y'] = resize_mat * [x, y]
    for n in range(len(jsl_data)):
        # Process left-top point to new position
        lt_pt = np.array([jsl_data[n]['left'], jsl_data[n]['top']])
        new_lt = [int(i) for i in (resize_mat * lt_pt)]    # Pixel position is {INT}
        # Pass new values
        jsl_out[n]['left'] = new_lt[0]
        jsl_out[n]['top'] = new_lt[1]
        if bb_meas is None:
            # Process right-bottom point to new position
            rb_pt = np.array([jsl_data[n]['left'] + jsl_data[n]['width'], jsl_data[n]['top'] + jsl_data[n]['height']])
            new_rb = [int(i) for i in (resize_mat * rb_pt)]    # Pixel position is {INT}
            # Calculate new width and height
            new_w = new_rb[0] - new_lt[0]
            if new_w < 1:    # Avoid 0 pixel value after transform
                new_w = 1
            new_h = new_rb[1] - new_lt[1]
            if new_h < 1:    # Avoid 0 pixel value after transform
                new_h = 1
            # Pass new values
            jsl_out[n]['width'] = new_w
            jsl_out[n]['height'] = new_h
        else:
            jsl_out[n]['width'] = bb_meas[jsl_out[n]['label']][0]
            jsl_out[n]['height'] = bb_meas[jsl_out[n]['label']][1]
    return jsl_out


def jsl_plot(jsl_data, img, color_list=None, annotate=False):
    """ Plot JSON label information to corresponding image.

    Args:
        jsl_data (list[dict]): List of dictionary with label info.
        img (np.ndarray): {2D, 3D} Image array, image corresponding to current label.
        color_list (dict[str, tuple[int, int, int]] or None): Dictionary of colors linked with label (default: None).
        annotate (bool): Defines if label texts will be on the converted image.

    Returns:
        np.ndarray: {3D} Image array, image with labels plotted on.
    """
    img_out = img    # Make copy of input, avoid unexpected modification
    curr_lbl = str()    # INIT VAR
    color = (255, 255, 255)    # INIT VAR
    for lbl in jsl_data:
        if (lbl['left'] is not None) or (lbl['top'] is not None):
            if color_list is None:
                if lbl['label'] != curr_lbl:    # Generate random colors for different label
                    color = rand_color(mode='RGB', norm=False)
                    curr_lbl = lbl['label']
            else:    # Use defined colors if color_list exists
                color = color_list[lbl['label']][::-1]    # Reverse RGB order for OpenCV BGR mode
            lt_pt = (lbl['left'], lbl['top'])    # Left-Top position
            rb_pt = (lbl['left'] + lbl['width'], lbl['top'] + lbl['height'])    # Right-Bottom position
            cv2.rectangle(img=img_out, pt1=lt_pt, pt2=rb_pt, color=color, thickness=1)    # Plot bounding box
            if annotate:
                lbl_pt = (rb_pt[0] + 2, rb_pt[1])    # Annotation position
                cv2.putText(img=img_out, text=lbl['label'], org=lbl_pt, fontFace=0, fontScale=0.3, color=color)
                # cv2.putText::fontFace=0: FONT_HERSHEY_SIMPLEX
        else:
            print("Label [%s] missing!" % lbl['label'])
    return img_out


# Label conversion functions ----------------------------------------------------------------------------------------- #

def lblconv_json2heat(jsl_data, img, bb_meas=None, peak=1.0):
    """ Convert JSON type label to HeatMap type label.

    Args:
        jsl_data (list[dict]): List of dictionary with JSON type label info.
        img (np.ndarray): {2D, 3D} Image array, image corresponding to the label.
        bb_meas (dict[str, tuple[int, int]] or None): Define bounding box size for each type of labels (default: None).
                --  KEY (str) = label name
                --  VALUE (tuple[int, int]) = (width, height)
        peak (int or float): Peak value of generated HeatMap (default: 16.0).

    Returns:
        list[dict]: List of dictionary with HeatMap type label info.
                --  KEY (str) = label name
                --  VALUE (np.ndarray) = {2D} Normalized PDF as label heatmap.
    """
    # Get plot region
    lim_y, lim_x = img.shape[:2]
    x = np.arange(start=0, stop=lim_x, step=1, dtype=np.uint32)
    y = np.arange(start=0, stop=lim_y, step=1, dtype=np.uint32)
    # Get plot mesh
    xx, yy = np.meshgrid(x, y)
    xxyy = np.c_[xx.ravel(), yy.ravel()]

    # Compute JSON label data to HeatMap label data
    hml_data = []    # INIT VAR
    lbl_temp = {'label': None, 'heatmap': None}    # INIT VAR
    for lbl in jsl_data:
        if (lbl['left'] is None) or (lbl['top'] is None):
            heatmap = np.zeros((lim_y, lim_x), dtype=np.float64)    # Handel dummy labels
        else:
            # Get center value of original label
            center_x = lbl['left'] + lbl['width'] / 2
            center_y = lbl['top'] + lbl['height'] / 2
            # Get bounding box size
            if bb_meas is None:
                width = lbl['width']
                height = lbl['height']
            else:
                width = bb_meas[lbl['label']][0]
                height = bb_meas[lbl['label']][1]

            # Compute heatmap from 2D-Gaussian PDF
            mean = [center_x, center_y]
            cov = [[width, 0], [0, height]]
            pdf = stats.multivariate_normal.pdf(xxyy, mean=mean, cov=cov)
            # Reshape, condition(value trim), and normalize PDF
            heatmap = pdf.reshape((lim_y, lim_x))    # Reshape
            trim_lim = 0.1 * pdf.max()
            heatmap = np.where(heatmap < trim_lim, 0, heatmap)    # Trim
            hm_peak = peak / np.max(heatmap, axis=None)
            heatmap = np.multiply(heatmap, hm_peak)    # Transform max value to peak value

        # Sending data
        lbl_temp['label'] = lbl['label']
        lbl_temp['heatmap'] = heatmap
        hml_data.append(copy.deepcopy(lbl_temp))

    return hml_data


def lblconv_heat2json(hml_data, th):
    """ Convert HeatMap type label to JSON type label.

    Args:
        hml_data (list[dict]}: List of dictionary with HeatMap type label info.
        th (int or float): HeatMap peak value threshold.

    Returns:
        list[dict]: List of dictionary with JSON label info.
    """
    jsl_data = []  # INIT VAR
    jsl_temp = {'left': None, 'top': None, 'width': 1, 'height': 1, 'label': None}    # INIT VAR
    js_dummy = {'left': None, 'top': None, 'width': 0, 'height': 0, 'label': None}    # INIT VAR
    for lbl in hml_data:
        flag = True    # INIT/RESET VAR
        if lbl['heatmap'] is not None:
            if lbl['heatmap'].max() > th:
                flag = False    # Valid HeatMap
                jsl_temp['label'] = lbl['label']
                top, left = np.unravel_index(np.argmax(lbl['heatmap']), lbl['heatmap'].shape)
                jsl_temp['top'] = top.item()
                jsl_temp['left'] = left.item()
                jsl_data.append(copy.deepcopy(jsl_temp))
        if flag:
            js_dummy['label'] = lbl['label']
            jsl_data.append(copy.deepcopy(js_dummy))
    return jsl_data


# HeatMap type label file functions ---------------------------------------------------------------------------------- #

def hml_read(hm_lbl_file):
    """ Import a PICKLE HeatMap label file to a Python list of dictionary.

    Args:
        hm_lbl_file (str): Labelling file contained with HeatMap labels.

    Returns:
        list[dict]: List of dictionary with HeatMap label info.
    """
    with open(hm_lbl_file, 'rb') as infile:
        comp = pkl.load(infile)
    hml_data = pkl.loads(zlib.decompress(comp))
    return hml_data


def hml_write(dst_hml_file, hml_data):
    """ Write HeatMap label data to a PICKLE label file.

    Args:
        dst_hml_file (str): Labelling file to write HeatMap labels (*.pkl).
        hml_data (list[dict]): List of dictionary with HeatMap label info.

    Returns:
        bool: File creation status.
    """
    if len(hml_data) != 0:
        comp = zlib.compress(pkl.dumps(hml_data, protocol=2))
        with open(dst_hml_file, 'wb') as outfile:
            pkl.dump(comp, outfile, protocol=2)
        return True
    else:
        print('Empty label data input, file not created!')
        return False


def hml_verify(hml_data, lbl_list, hm_dummy=None):
    """ Verify HeatMap type labels, padding missing labels, standardize label order.

    Args:
        hml_data (list[dict]): List of dictionary with HeatMap label info.
        lbl_list (list[str]): List of all label tags used in labelling, including duplicates, order sensitive.
        hm_dummy (np.ndarray or None): {2D} HeatMap used to override when some label is missing (default: None).

    Returns:
        tuple[list[dict], bool]:
            hml_out (list[dict]): List of dictionary with verified and standardized HeatMap label info.
            has_dum (bool): A flag to identify whether [hml_out] has dummy data insertion or not.
    """
    lbl_copy = copy.deepcopy(hml_data)    # Make copy of original data, avoid unexpected modification
    lbl_temp = {'label': None, 'heatmap': None}    # INIT VAR
    hml_out = []    # INIT VAR
    has_dum = False    # INIT VAR, False by default, no dummy data inserted.
    for lbl_name in lbl_list:    # Checking labels in defined sequence
        flag = True    # RESET VAR: (label missing) flag to True (not found yet)
        for lbl in lbl_copy:    # Label searching
            if lbl['label'] == lbl_name:
                lbl_temp = copy.deepcopy(lbl)
                lbl_copy.remove(lbl)    # Avoid getting same label multiple times
                flag = False    # Label found, (label missing) flag set to False
                break    # Avoid pass-over when found
        if flag:    # Padding missing label with dummy
            lbl_temp['label'] = lbl_name
            lbl_temp['heatmap'] = hm_dummy
            has_dum = True    # Set status to True as dummy data inserted for missing data.
        hml_out.append(copy.deepcopy(lbl_temp))    # Putting labels in defined order
    return hml_out, has_dum


def hml_plot(hml_data, img, color_list=None):
    """ Plot HeatMap label information to corresponding image.

    Args:
        hml_data (list[dict]): List of dictionary with HeatMap type label info.
        img (np.ndarray): {2D, 3D} Image array, image corresponding to current label.
        color_list (dict[str, tuple[int, int, int]] or None): Dictionary of colors linked with label (default: None).

    Returns:
        np.ndarray: {3D} Image array, image with labels plotted on.
    """
    merged_map = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)    # INIT VAR
    if color_list is None:
        for lbl in hml_data:
            if lbl['heatmap'] is not None:
                if lbl['heatmap'].max() > 0:    # Exclude negative HeatMaps
                    # Clip for negative values
                    hm = np.clip(lbl['heatmap'], 0, None)
                    # Re-normalize heatmap to [0, 255] UINT8 as color map base
                    hm_norm = np.multiply(hm, 255 / hm.max()).astype(np.uint8)
                    # Creat a color map with given heatmap values
                    heat_img = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)
                    heat_img[np.where((heat_img == [128, 0, 0]).all(axis=2))] = [0, 0, 0]    # Set 0-HEAT as transparent
                    # Merge heatmaps of labels
                    merged_map = cv2.add(merged_map, heat_img)
        # Overlay heatmap to original image
        img_out = cv2.addWeighted(img, 1, merged_map, 0.75, 0)
    else:
        for lbl in hml_data:
            if lbl['heatmap'] is not None:
                if lbl['heatmap'].max() > 0:    # Exclude negative HeatMaps
                    # Clip for negative values
                    hm = np.clip(lbl['heatmap'], 0, None)
                    # Re-normalize heatmap to [0, 1] FLOAT64 as alpha level base
                    hm_norm = np.multiply(hm, 1 / hm.max())
                    # Assign defined color with heatmap as alpha level
                    color = color_list[lbl['label']][::-1]    # Reverse RGB order for OpenCV BGR mode
                    heat_img = np.dstack((hm_norm * color[0], hm_norm * color[1], hm_norm * color[2])).astype(np.uint8)
                    # Merge heatmaps of labels
                    merged_map = cv2.add(merged_map, heat_img)
        # Overlay heatmap to original image
        img_out = cv2.addWeighted(img, 1, merged_map, 1.5, 0)
    return img_out


def hml_jntplot(hml_data, img, joint, color_list=None):
    """ Plot defined HeatMap label names to corresponding image.

    Args:
        hml_data (list[dict]): List of dictionary with HeatMap type label info.
        img (np.ndarray): {2D, 3D} Image array, image corresponding to current label.
        joint (str or list[str]): Name of the joint(s) to be plotted.
        color_list (dict[str, tuple[int, int, int]] or None): Dictionary of colors linked with label (default: None).

    Returns:
        np.ndarray: {3D} Image array, image with labels plotted on.
    """
    merged_map = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)    # INIT VAR
    if color_list is None:
        for lbl in hml_data:
            if (lbl['heatmap'] is not None) and (lbl['label'] in joint):
                if lbl['heatmap'].max() > 0:    # Exclude negative HeatMaps
                    # Clip for negative values
                    hm = np.clip(lbl['heatmap'], 0, None)
                    # Re-normalize heatmap to [0, 255] UINT8 as color map base
                    hm_norm = np.multiply(hm, 255 / hm.max()).astype(np.uint8)
                    # Creat a color map with given heatmap values
                    heat_img = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)
                    heat_img[np.where((heat_img == [128, 0, 0]).all(axis=2))] = [0, 0, 0]    # Set 0-HEAT as transparent
                    # Merge heatmaps of labels
                    merged_map = cv2.add(merged_map, heat_img)
        # Overlay heatmap to original image
        img_out = cv2.addWeighted(img, 1, merged_map, 0.75, 0)
    else:
        for lbl in hml_data:
            if (lbl['heatmap'] is not None) and (lbl['label'] in joint):
                if lbl['heatmap'].max() > 0:    # Exclude negative HeatMaps
                    # Clip for negative values
                    hm = np.clip(lbl['heatmap'], 0, None)
                    # Re-normalize heatmap to [0, 1] FLOAT64 as alpha level base
                    hm_norm = np.multiply(hm, 1 / hm.max())
                    # Assign defined color with heatmap as alpha level
                    color = color_list[lbl['label']][::-1]    # Reverse RGB order for OpenCV BGR mode
                    heat_img = np.dstack((hm_norm * color[0], hm_norm * color[1], hm_norm * color[2])).astype(np.uint8)
                    # Merge heatmaps of labels
                    merged_map = cv2.add(merged_map, heat_img)
        # Overlay heatmap to original image
        img_out = cv2.addWeighted(img, 1, merged_map, 1.5, 0)
    return img_out


def hm_optical_warp(img_prev, img_next, hm_tensor):
    """ Compute HeatMap shift between previous image and next image using optical-flow.

    Args:
        img_prev (np.ndarray): {1D} Previous image, also is the target image, in Gray-scale.
        img_next (np.ndarray): {1D} Previous image, also is the reference image, in Gray-scale.
        hm_tensor (np.ndarray): {4D} HeatMap label tensor corresponding to [img_prev].

    Returns:
        np.ndarray: {4D} Optical flow warped HeatMap tensor.
    """
    # Compute optical-flow array
    img_prev = cv2.cvtColor(img_prev, cv2.COLOR_BGR2GRAY)
    img_next = cv2.cvtColor(img_next, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(img_prev, img_next, flow=None, pyr_scale=0.5, levels=4, iterations=8,
                                        winsize=27, poly_n=7, poly_sigma=1.5, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    # From optical-flow create remap array
    height, width = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] += np.arange(width)
    flow[:, :, 1] += np.arange(height)[:, np.newaxis]
    # Execute optical-flow warp to heat-map
    lbl_temp = []  # INIT VAR
    for i in range(hm_tensor.shape[2]):
        lbl = hm_tensor[:, :, i]
        wrp = cv2.remap(lbl, flow, None, cv2.INTER_LINEAR)
        lbl_temp.append(wrp)
    hm_wrp = np.stack(lbl_temp, axis=-1)
    return hm_wrp


def hm_multi_warp(img_lst, hm_tensor):
    """ Compute HeatMap shift between a sequence of images using optical-flow.

    Args:
        img_lst (list[np.ndarray]): {LST od 1D} List of gray-scale images, warp HeatMap with defined sequence.
        hm_tensor (np.ndarray): {4D} HeatMap label tensor corresponding to [img_lst[0]].

    Returns:
        np.ndarray: {4D} Optical flow warped HeatMap tensor.
    """
    hm_wrp = hm_tensor    # INIT VAR
    for i in range(len(img_lst) - 1):
        hm_wrp = hm_optical_warp(img_lst[i], img_lst[i + 1], hm_wrp)
    return hm_wrp


# Label batch process functions -------------------------------------------------------------------------------------- #

def lbl_merge(lbl_file, lbl_list):
    """ Read and merge multiple JSON label files into 2 NumPy nD-arrays.

    Args:
        lbl_file (list[str]): List of label files to be merged into an nD-array.
        lbl_list (list[str]): List of label names to be used.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            lbl_frm (np.ndarray): {1D} Array of frame numbers.
            lbl_crv_x (np.ndarray): Array of label LEFT positions, arranged with [lbl_list] order.
            lbl_crv_y (np.ndarray): Array of label TOP positions, arranged with [lbl_list] order.

    """
    lbl_frm = np.arange(start=1, stop=len(lbl_file) + 1, step=1, dtype=int)    # Get output X
    lbl_crv_x = np.empty((len(lbl_list), len(lbl_file)), dtype=int)    # INIT VAR
    lbl_crv_y = np.empty((len(lbl_list), len(lbl_file)), dtype=int)    # INIT VAR
    for i in range(len(lbl_file)):
        lbl_data = jsl_read(lbl_file[i])
        for j in range(len(lbl_list)):
            flag = True    # RESET VAR: (label missing) flag to True (not found yet)
            for lbl in lbl_data:
                if lbl['label'] == lbl_list[j]:
                    lbl_crv_x[j][i] = lbl['left']
                    lbl_crv_y[j][i] = lbl['top']
                    lbl_data.remove(lbl)    # Avoid getting same label multiple times
                    flag = False    # Label found, (label missing) flag set to False
                    break    # Avoid pass-over when found
            if flag:
                print("Label [%s] in file [%s] missing!" % (lbl_list[j], lbl_file[i]))
    return lbl_frm, lbl_crv_x, lbl_crv_y


def lbl_split(lbl_crv_x, lbl_crv_y, lbl_file, lbl_list, bb_meas=None):
    """ Split and write 2 NumPy nD-arrays back to JSON file.

    Args:
        lbl_crv_x (np.ndarray): Array of label LEFT positions, arranged with [lbl_list] order.
        lbl_crv_y (np.ndarray): Array of label TOP positions, arranged with [lbl_list] order.
        lbl_file (list[str]): List of label files to be merged into an nD-array.
        lbl_list (list[str]): List of label names to be used.
        bb_meas (dict[str, tuple[int, int]] or None): Define bounding box size for each type of labels (default: None).
                --  KEY (str) = label name
                --  VALUE (tuple[int, int]) = (width, height)

    Returns:
    """
    count_i = lbl_crv_x.shape[1]
    count_j = len(lbl_list)
    lbl_temp = {'left': None, 'top': None, 'width': None, 'height': None, 'label': None}    # INIT VAR
    for i in range(count_i):
        lbl = []    # INIT/RESET VAR
        for j in range(count_j):
            lbl_temp['label'] = lbl_list[j]
            lbl_temp['left'] = lbl_crv_x[j][i].item()
            lbl_temp['top'] = lbl_crv_y[j][i].item()
            if bb_meas is None:
                lbl_temp['width'] = lbl_temp['height'] = 1
            else:
                lbl_temp['width'], lbl_temp['height'] = bb_meas[lbl_list[j]]
            lbl.append(copy.deepcopy(lbl_temp))
        jsl_write(lbl_file[i], lbl)
