import copy
import numpy as np
import cv2
from OptiFlex.utils.base_func import os_rand_range

"""Function list:
# Image only functions:
    img_bkgsub(src, bkg):  Simple background subtraction with an image of background alone.
    img_trh_nobkg(src, c, ksize):  A semi-adaptive threshold method for background subtracted images.
# Image / JSON-label functions:
    imjsl_crop(img, jsl, size, origin):  Crop image array with its corresponding JSON labels.
    imjsl_resize(img, jsl, size, interpolation):  Resize image array with its corresponding JSON labels.
    imjsl_flip(img, jsl, axis):  Flip image array with its corresponding JSON labels.
    imjsl_rotate(img, jsl, angle):  Rotate image array with its corresponding JSON labels.
    imjsl_aug(img, jsl, size, axis, angle): Random process an image with its JSON labels for data augmentation.
# Image / HeatMap-label functions:
    imhml_crop(img, hml, size, origin):  Crop image array with its corresponding HeatMap labels.
    imhml_resize(img, hml, size, interpolation):  Resize image array with its corresponding HeatMap labels.
    imhml_flip(img, hml, axis):  Flip image array with its corresponding HeatMap labels.
    imhml_rotate(img, hml, angle):  Rotate image array with its corresponding HeatMap labels.
    imhml_aug(img, hml, size, axis, angle):  Data augmentation for an image with its HeatMap labels.
"""


# Image only functions ----------------------------------------------------------------------------------------------- #

def img_bkgsub(src, bkg):
    """ Simple background subtraction with an image of background alone.

    Args:
        src (np.ndarray): {2D, 3D} Image array, image to be processed.
        bkg (np.ndarray): {2D, 3D} Background image array, MUST be same size/type as [src].

    Returns:
        np.ndarray: {2D, 3D} Image array, image with background subtracted.
    """
    # Subtract background
    dst = cv2.subtract(src, bkg)
    # Reduce typical salt-and-pepper noise
    dst = cv2.medianBlur(dst, ksize=3)
    return dst


def img_trh_nobkg(src, c=60, ksize=3):
    """ A semi-adaptive threshold method for background subtracted images.

    Args:
        src (np.ndarray): {2D} Image array, must be signal channel, image to be processed.
        c (int): A constant used to control auto-thresh value and thresh lower limit.
        ksize (int): Kernel size for median filter.

    Returns:
        np.ndarray: {2D-BIN} Image array, threshold/filtered image.
    """
    th = (src.max() >> 1) - c    # Get thresh value
    th = c if th < c else th    # Limit thresh value
    _, dst = cv2.threshold(src, th, 255, cv2.THRESH_BINARY)
    dst = cv2.medianBlur(dst, ksize=ksize)    # Removing small pixels
    return dst


# Image / JSON-label functions --------------------------------------------------------------------------------------- #

def imjsl_crop(img, jsl, size, origin=(0, 0)):
    """ Crop image array with its corresponding JSON labels.

    Args:
        img (np.ndarray): {2D, 3D} Image array, image to be processed.
        jsl (list[dict]): Array of dictionary with label info to be processed.
        size (tuple[int, int]): Defines final size of image (in pixel):
                --  size[0] = w, new image width
                --  size[1] = h, new image height
        origin (tuple[int, int]): Defines crop origin of image (in pixel, default: (0, 0)):
                --  origin[0] = x, left pixel position for crop
                --  origin[1] = y, top pixel position for crop

    Returns:
        tuple[np.ndarray, list[dict]]:
            img_out (np.ndarray): {2D, 3D} Image array, cropped image.
            jsl_out (list[dict]): Array of dictionary with cropped pixel position of labels.
    """
    # Get basic image info
    img_h, img_w = img.shape[:2]    # Image.shape {3-TUPLE}: height, width, color_space
    # Verify crop origin parameters
    left = origin[0]
    top = origin[1]
    if left < 0:
        print('Crop left origin less than 0, crop to 0.')
        left = 0
    if top < 0:
        print('Crop top origin less than 0, crop to 0.')
        top = 0
    # Calculate and verify crop endpoint parameters
    right = left + size[0]
    bottom = top + size[1]
    if right > img_w:
        print('Crop width exceeds original image, crop to original width.')
        right = img_w
    if bottom > img_h:
        print('Crop height exceeds original image, crop to original height.')
        bottom = img_h

    # Execute image crop
    img_out = img[top:bottom, left:right]
    # Execute label crop
    lbl_temp = {'left': None, 'top': None, 'width':  None, 'height':  None, 'label': None}    # INIT VAR
    jsl_out = []    # INIT VAR
    for lbl_data in jsl:
        # Calculate new label values
        lbl_left = lbl_data['left'] - left
        lbl_right = lbl_left + lbl_data['width']
        lbl_top = lbl_data['top'] - top
        lbl_bottom = lbl_top + lbl_data['height']
        # Verify new value for passing to output
        if (lbl_left < 0) or (lbl_top < 0) or (lbl_right > size[0]) or (lbl_bottom > size[1]):
            continue
        else:
            lbl_temp['left'] = lbl_left
            lbl_temp['top'] = lbl_top
            lbl_temp['width'] = lbl_data['width']
            lbl_temp['height'] = lbl_data['height']
            lbl_temp['label'] = lbl_data['label']
            jsl_out.append(copy.deepcopy(lbl_temp))

    return img_out, jsl_out


def imjsl_resize(img, jsl, size, interpolation=1):
    """ Resize image array with its corresponding JSON labels.

    Args:
        img (np.ndarray): {2D, 3D} Image array, image to be processed.
        jsl (list[dict]): Array of dictionary with label info to be processed.
        size (tuple[int, int]): Defines final size of image (in pixel):
                --  size[0] = new image width
                --  size[1] = new image height
        interpolation (int): {[0, 5]} OpenCV interpolation method (default: 1):
                --  0 = INTER_NEAREST - a nearest-neighbor interpolation
                --  1 = INTER_LINEAR - a bilinear interpolation
                --  2 = INTER_CUBIC - a bicubic interpolation over 4x4 pixel neighborhood
                --  3 = INTER_AREA - resampling using pixel area relation (Moire pattern free)
                --  4 = INTER_LANCZOS4 - a Lanczos interpolation over 8x8 pixel neighborhood
                --  5 = INTER_LINEAR_EXACT - a modified bilinear interpolation

    Returns:
        tuple[np.ndarray, list[dict]]:
            img_out (np.ndarray): {2D, 3D} Image array, resized image.
            jsl_out (list[dict]): Array of dictionary with resized pixel position of labels.
    """
    # Get basic image info
    img_h, img_w = img.shape[:2]    # Image.shape {3-TUPLE}: height, width, color_space

    # Execute image resize
    img_out = cv2.resize(src=img, dsize=size, interpolation=interpolation)
    # Execute label resize -- [x', y'] = resize_mat * [x, y]
    resize_mat = np.array([size[0]/img_w, size[1]/img_h])
    jsl_out = copy.deepcopy(jsl)    # INIT VAR
    for n in range(len(jsl)):
        # Process left-top point to new position
        lt_pt = np.array([jsl[n]['left'], jsl[n]['top']])
        new_lt = [int(i) for i in (resize_mat * lt_pt)]    # Pixel position is {INT}
        # Process right-bottom point to new position
        rb_pt = np.array([jsl[n]['left'] + jsl[n]['width'], jsl[n]['top'] + jsl[n]['height']])
        new_rb = [int(i) for i in (resize_mat * rb_pt)]    # Pixel position is {INT}
        # Pass new values
        jsl_out[n]['left'] = new_lt[0]
        jsl_out[n]['top'] = new_lt[1]
        new_w = new_rb[0] - new_lt[0]
        if new_w < 1:    # Avoid 0 pixel value after transform
            new_w = 1
        jsl_out[n]['width'] = new_w
        new_h = new_rb[1] - new_lt[1]
        if new_h < 1:    # Avoid 0 pixel value after transform
            new_h = 1
        jsl_out[n]['height'] = new_h

    return img_out, jsl_out


def imjsl_flip(img, jsl, axis=1):
    """ Flip image array with its corresponding JSON labels.

    Args:
        img (np.ndarray): {2D, 3D} Image array, image to be processed.
        jsl (list[dict]): Array of dictionary with label info to be processed.
        axis (int): {-1 OR 0 OR 1} OpenCV flag to specify how to flip the array (default: 1):
                --   1 = flipping around y-axis
                --   0 = flipping around the x-axis
                --  -1 = flipping around both axes

    Returns:
        tuple[np.ndarray, list[dict]]:
            img_out (np.ndarray): {2D, 3D} Image array, flipped image.
            jsl_out (list[dict]): Array of dictionary with flipped pixel position of labels.
    """
    # Get basic image info
    img_h, img_w = img.shape[:2]    # Image.shape {3-TUPLE}: height, width, color_space

    # Execute image flip
    img_out = cv2.flip(img, axis)
    # Execute label flip
    jsl_out = copy.deepcopy(jsl)    # INIT VAR
    if axis > 0:    # Flip @ y-axis -- x' = img_w - x - 1; y' = y
        for n in range(len(jsl)):
            jsl_out[n]['left'] = img_w - jsl[n]['left'] - jsl[n]['width'] - 1
    elif axis == 0:    # Flip @ x-axis -- x' = x; y' = img_h - y - 1
        for n in range(len(jsl)):
            jsl_out[n]['top'] = img_h - jsl[n]['top'] - jsl[n]['height'] - 1
    else:    # Flip @ x & y axes -- x' = img_w - x - 1; y' = img_h - y - 1
        for n in range(len(jsl)):
            jsl_out[n]['left'] = img_w - jsl[n]['left'] - jsl[n]['width'] - 1
            jsl_out[n]['top'] = img_h - jsl[n]['top'] - jsl[n]['height'] - 1

    return img_out, jsl_out


def imjsl_rotate(img, jsl, angle):
    """ Rotate image array with its corresponding JSON labels.

    Args:
        img (np.ndarray): {2D, 3D} Image array, image to be processed.
        jsl (list[dict]): Array of dictionary with label info to be processed.
        angle (float): Defines rotation angle of image (in degree):
                --  > 0 = CCW
                --  < 0 = CW

    Returns:
        tuple[np.ndarray, list[dict]]:
            img_out (np.ndarray): {2D, 3D} Image array, rotated image.
            jsl_out (list[dict]): Array of dictionary with rotated pixel position of labels.
    """
    # Get basic image info
    img_h, img_w = img.shape[:2]    # Image.shape {3-TUPLE}: height, width, color_space
    center = (img_w/2, img_h/2)

    # Creat rotation matrix
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1)    # scale = 1, no resize
    # Get absolute value of rotation cos and sin
    abs_cos = abs(rot_mat[0, 0])
    abs_sin = abs(rot_mat[0, 1])
    # Calculate new image bounds
    bnd_w = int(img_h * abs_sin + img_w * abs_cos)
    bnd_h = int(img_h * abs_cos + img_w * abs_sin)
    # Recenter rotated image
    rot_mat[0, 2] += bnd_w/2 - center[0]
    rot_mat[1, 2] += bnd_h/2 - center[1]

    # Execute image rotation
    img_out = cv2.warpAffine(img, rot_mat, (bnd_w, bnd_h))
    # Execute label rotation -- cv2::getAffineTransform: [x'; y'] = rot_map (DOT) [x; y; 1]
    jsl_out = copy.deepcopy(jsl)    # INIT VAR
    for n in range(len(jsl)):
        # Process left-top point to new position
        lt_pt = np.array([[jsl[n]['left']], [jsl[n]['top']], [1]])
        new_lt = [int(i) for i in np.dot(rot_mat, lt_pt)]    # Pixel position is {INT}
        # Process right-bottom point to new position
        rb_pt = np.array([[jsl[n]['left'] + jsl[n]['width']], [jsl[n]['top'] + jsl[n]['height']], [1]])
        new_rb = [int(i) for i in np.dot(rot_mat, rb_pt)]    # Pixel position is {INT}
        # Pass new values
        jsl_out[n]['left'] = new_lt[0]
        jsl_out[n]['top'] = new_lt[1]
        new_w = new_rb[0] - new_lt[0]
        if new_w < 1:    # Avoid 0 pixel value after transform
            new_w = 1
        jsl_out[n]['width'] = new_w
        new_h = new_rb[1] - new_lt[1]
        if new_h < 1:    # Avoid 0 pixel value after transform
            new_h = 1
        jsl_out[n]['height'] = new_h

    return img_out, jsl_out


def imjsl_aug(img, jsl, size, axis, angle, rnd_flp=True, rnd_rot=True):
    """ Random process an image with its JSON labels for data augmentation.

    Args:
        img (np.ndarray): {2D, 3D} Image array, image to be processed.
        jsl (list[dict]): Array of dictionary with label info to be processed.
        size (tuple[int, int]): Defines final size of image (in pixel):
                --  size[0] = new image width
                --  size[1] = new image height
        axis (list[int]): {-1 AND-OR 0 AND-OR 1} Specify how to flip the array (randomized in choice):
                --   1 = flipping around y-axis
                --   0 = flipping around the x-axis
                --  -1 = flipping around both axes
        angle (tuple[float, float]): Defines rotation angle of image (randomized in range).
                --  > 0 = CCW
                --  < 0 = CW
        rnd_flp (bool): Control if the random flipping will be used in augmentation (default: True).
        rnd_rot (bool): Control if the random rotation will be used in augmentation (default: True).

    Returns:
        tuple[np.ndarray, list[dict]]:
            img (np.ndarray): {2D, 3D} Image array, random transformed image.
            jsl (list[dict]): Array of dictionary with random transformed labels (same as [img]).
    """
    # Get random session
    flip = np.random.choice([True, False])
    rotate = np.random.choice([True, False])

    # Process augmentation codes
    if rnd_flp and flip:    # Random flip execution
        rand_axis = np.random.choice(axis)    # Flipping axis choice
        img, jsl = imjsl_flip(img, jsl, rand_axis)
    if rnd_rot and rotate:    # Random rotate execution
        rand_angle = os_rand_range(angle[0], angle[1], size=8, digits=2)    # Cryptographically secure pseudo-random
        img, jsl = imjsl_rotate(img, jsl, rand_angle)
    # Always resize to the same size
    img, jsl = imjsl_resize(img, jsl, size, 3)    # interpolation = 3: INTER_AREA

    return img, jsl


# Image / HeatMap-label functions ------------------------------------------------------------------------------------ #

def imhml_crop(img, hml, size, origin=(0, 0)):
    """ Crop image array with its corresponding HeatMap labels.

    Args:
        img (np.ndarray): {2D, 3D} Image array, image to be processed.
        hml (list[dict]): Array of dictionary with HeatMap type label info to be processed.
        size (tuple[int, int]): Defines final size of image (in pixel):
                --  size[0] = w, new image width
                --  size[1] = h, new image height
        origin (tuple[int, int]): Defines crop origin of image (in pixel, default: (0, 0)):
                --  origin[0] = x, left pixel position for crop
                --  origin[1] = y, top pixel position for crop

    Returns:
        tuple[np.ndarray, list[dict]]:
            img_out (np.ndarray): {2D, 3D} Image array, cropped image.
            hml_out (list[dict]): Array of dictionary with cropped heatmap of labels.
    """
    # Get basic image info
    img_h, img_w = img.shape[:2]    # Image.shape {3-TUPLE}: height, width, color_space
    # Verify crop origin parameters
    left = origin[0]
    top = origin[1]
    if left < 0:
        print('Crop left origin less than 0, crop to 0.')
        left = 0
    if top < 0:
        print('Crop top origin less than 0, crop to 0.')
        top = 0
    # Calculate and verify crop endpoint parameters
    right = left + size[0]
    bottom = top + size[1]
    if right > img_w:
        print('Crop width exceeds original image, crop to original width.')
        right = img_w
    if bottom > img_h:
        print('Crop height exceeds original image, crop to original height.')
        bottom = img_h

    # Execute image crop
    img_out = img[top:bottom, left:right]
    # Execute label crop with the same method
    hml_out = copy.deepcopy(hml)    # INIT VAR
    for lbl in hml_out:
        if lbl['heatmap'] is not None:
            lbl['heatmap'] = lbl['heatmap'][top:bottom, left:right]

    return img_out, hml_out


def imhml_resize(img, hml, size, interpolation=1, peak=16.):
    """ Resize image array with its corresponding HeatMap labels.

    Args:
        img (np.ndarray): {2D, 3D} Image array, image to be processed.
        hml (list[dict]): Array of dictionary with HeatMap type label info to be processed.
        size (tuple[int, int]): Defines final size of image (in pixel):
                --  size[0] = new image width
                --  size[1] = new image height
        interpolation (int): {[0, 5]} OpenCV interpolation method (default: 1):
                --  0 = INTER_NEAREST - a nearest-neighbor interpolation
                --  1 = INTER_LINEAR - a bilinear interpolation
                --  2 = INTER_CUBIC - a bicubic interpolation over 4x4 pixel neighborhood
                --  3 = INTER_AREA - resampling using pixel area relation (Moire pattern free)
                --  4 = INTER_LANCZOS4 - a Lanczos interpolation over 8x8 pixel neighborhood
                --  5 = INTER_LINEAR_EXACT - a modified bilinear interpolation
        peak (int or float): Peak value of transformed HeatMap (default: 16.0).

    Returns:
        tuple[np.ndarray, list[dict]]:
            img_out (np.ndarray): {2D, 3D} Image array, resized image.
            hml_out (list[dict]): Array of dictionary with resized heatmap of labels.
    """
    # Resize image
    img_out = cv2.resize(img, size, interpolation)

    # Resize HeatMap label with the same method
    hml_out = copy.deepcopy(hml)
    for lbl in hml_out:
        if lbl['heatmap'] is not None:
            lbl['heatmap'] = cv2.resize(lbl['heatmap'], size, interpolation)    # Resize using OpenCV library
            hm_max = np.max(lbl['heatmap'], axis=None)
            if hm_max > 0:
                lbl['heatmap'] = np.multiply(lbl['heatmap'], peak / hm_max)    # Re-peak

    return img_out, hml_out


def imhml_flip(img, hml, axis=1, peak=16.):
    """ Flip image array with its corresponding HeatMap labels.

    Args:
        img (np.ndarray): {2D, 3D} Image array, image to be processed.
        hml (list[dict]): Array of dictionary with HeatMap type label info to be processed.
        axis (int): {-1 OR 0 OR 1} OpenCV flag to specify how to flip the array (default: 1):
                --   1 = flipping around y-axis
                --   0 = flipping around the x-axis
                --  -1 = flipping around both axes
        peak (int or float): Peak value of transformed HeatMap (default: 16.0).

    Returns:
        tuple[np.ndarray, list[dict]]:
            img_out (np.ndarray): {2D, 3D} Image array, flipped image.
            hml_out (list[dict]): Array of dictionary with flipped heatmap of labels.
    """
    # Execute image flip
    img_out = cv2.flip(img, axis)
    # Execute label flip with the same method
    hml_out = copy.deepcopy(hml)    # INIT VAR
    for lbl in hml_out:
        if lbl['heatmap'] is not None:
            lbl['heatmap'] = cv2.flip(lbl['heatmap'], axis)    # Flip using OpenCV library
            hm_max = np.max(lbl['heatmap'], axis=None)
            if hm_max > 0:
                lbl['heatmap'] = np.multiply(lbl['heatmap'], peak / hm_max)    # Re-peak

    return img_out, hml_out


def imhml_rotate(img, hml, angle, peak=16.):
    """ Rotate image array with its corresponding HeatMap labels.

    Args:
        img (np.ndarray): {2D, 3D} Image array, image to be processed.
        hml (list[dict]): Array of dictionary with HeatMap type label info to be processed.
        angle (float): Defines rotation angle of image (in degree):
                --  > 0 = CCW
                --  < 0 = CW
        peak (int or float): Peak value of transformed HeatMap (default: 16.0).

    Returns:
        tuple[np.ndarray, list[dict]]:
            img_out (np.ndarray): {2D, 3D} Image array, rotated image.
            hml_out (list[dict]): Array of dictionary with rotated heatmap of labels.
    """
    # Get basic image info
    img_h, img_w = img.shape[:2]    # Image.shape {3-TUPLE}: height, width, color_space
    center = (img_w/2, img_h/2)

    # Creat rotation matrix
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1)    # scale = 1, no resize
    # Get absolute value of rotation cos and sin
    abs_cos = abs(rot_mat[0, 0])
    abs_sin = abs(rot_mat[0, 1])
    # Calculate new image bounds
    bnd_w = int(img_h * abs_sin + img_w * abs_cos)
    bnd_h = int(img_h * abs_cos + img_w * abs_sin)
    # Recenter rotated image
    rot_mat[0, 2] += bnd_w/2 - center[0]
    rot_mat[1, 2] += bnd_h/2 - center[1]

    # Execute image rotation
    img_out = cv2.warpAffine(img, rot_mat, (bnd_w, bnd_h))
    # Execute label rotation with the same method
    hml_out = copy.deepcopy(hml)    # INIT VAR
    for lbl in hml_out:
        if lbl['heatmap'] is not None:
            lbl['heatmap'] = cv2.warpAffine(lbl['heatmap'], rot_mat, (bnd_w, bnd_h))    # Rotate using OpenCV library
            hm_max = np.max(lbl['heatmap'], axis=None)
            if hm_max > 0:
                lbl['heatmap'] = np.multiply(lbl['heatmap'], peak / hm_max)    # Re-peak

    return img_out, hml_out


def imhml_aug(img, hml, size, axis, angle, peak=16., rnd_flp=True, rnd_rot=True):
    """ Random process an image with its HeatMap labels for data augmentation.

    Args:
        img (np.ndarray): {2D, 3D} Image array, image to be processed.
        hml (list[dict]): Array of dictionary with HeatMap type label info to be processed.
        size (tuple[int, int]): Defines final size of image (in pixel):
                --  size[0] = new image width
                --  size[1] = new image height
        axis (list[int]): {-1 AND-OR 0 AND-OR 1} Specify how to flip the array (randomized in choice):
                --   1 = flipping around y-axis
                --   0 = flipping around the x-axis
                --  -1 = flipping around both axes
        angle (tuple[float, float]): Defines rotation angle of image (randomized in range).
                --  > 0 = CCW
                --  < 0 = CW
        peak (int or float): Peak value of transformed HeatMap (default: 16.0).
        rnd_flp (bool): Control if the random flipping will be used in augmentation (default: True).
        rnd_rot (bool): Control if the random rotation will be used in augmentation (default: True).

    Returns:
        tuple[np.ndarray, list[dict]]:
            img (np.ndarray): {2D, 3D} Image array, random transformed image.
            hml (list[dict]): Array of dictionary with random transformed heatmap labels (same as [img]).
    """
    # Get random session
    flip = np.random.choice([True, False])
    rotate = np.random.choice([True, False])

    # Process augmentation codes
    if rnd_flp and flip:    # Random flip execution
        rand_axis = np.random.choice(axis)    # Flipping axis choice
        img, hml = imhml_flip(img, hml, rand_axis, peak)
    if rnd_rot and rotate:    # Random rotate execution
        rand_angle = os_rand_range(angle[0], angle[1], size=8, digits=2)    # Cryptographically secure pseudo-random
        img, hml = imhml_rotate(img, hml, rand_angle, peak)
    # Always resize to the same size
    img, hml = imhml_resize(img, hml, size, 3)    # interpolation = 3: INTER_AREA

    return img, hml
