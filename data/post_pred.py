import os
import copy
import numpy as np
from skimage import io as ski_io
from skimage import feature as ski_ft
import matplotlib.pyplot as plt
from data.lbl_proc import hml_read, hml_jntplot

"""Function list:
hmpred_unravel(hm_pred):  Get possible feature coordinates form predicted HeatMap file.
disp_unravel(im_file, hm_file, n, show_overlay):  Display unraveled HeatMap predictions for human verification.
btm_mirror_paw_fix(btm_hm_pred, sd_hm_pred):  Correct animal paw switch issue for Bottom-Mirror type setup.
"""


def hmpred_unravel(hm_pred):
    """ Get possible feature coordinates form predicted HeatMap file.

    Args:
        hm_pred (list[dict]): List of dictionary with HeatMap label info.

    Returns:
        tuple[dict, dict]:
            abs_max (dict): {2D, 3D} Image array, flipped image.
                --  KEY (str) = label name
                --  VALUE (tuple[np.int64, np.int64]) = (x, y)
            loc_max (dict): Array of dictionary with flipped heatmap of labels.
                --  KEY (str) = label name
                --  VALUE (np.ndarray) = {2D} matrix of (x, y)
    """
    abs_max = {}    # INIT VAR
    loc_max = {}    # INIT VAR
    for hm in hm_pred:
        if hm['heatmap'].max() < 0:
            abs_max[hm['label']] = ()
        else:
            abs_max[hm['label']] = np.unravel_index(hm['heatmap'].argmax(), hm['heatmap'].shape)
        loc_max[hm['label']] = ski_ft.peak_local_max(hm['heatmap'], min_distance=5, threshold_abs=.1, threshold_rel=.5)
    return abs_max, loc_max


def disp_unravel(im_file, hm_file, n, show_overlay=False):
    """ Display unraveled HeatMap predictions for human verification.

    Args:
        im_file (str): Original image file.
        hm_file (str): Predicted HeatMap file.
        n (int): Index of label.
        show_overlay (bool): Flag to determine if original-heatmap overlay will be displayed (default: False).

    Returns:
    """
    # Read files
    im = ski_io.imread(im_file)
    hm_lbl = hml_read(hm_file)
    # Get required heatmap
    hm = hm_lbl[n]['heatmap']
    im_hm = hml_jntplot(hm_lbl, hm_lbl[n]['label'], im)
    # Unravel predictions
    abs_max, loc_max = hmpred_unravel(hm_lbl)
    coord_abs = abs_max[hm_lbl[n]['label']]
    coord_loc = loc_max[hm_lbl[n]['label']]

    # Plot major layout setup
    if show_overlay:
        fig, ax = plt.subplots(3, 3)
    else:
        fig, ax = plt.subplots(3, 2)
    fig.suptitle(('%s -- %s' % (os.path.splitext(os.path.split(im_file)[1])[0], hm_lbl[n]['label'])), fontsize=16)
    fig.tight_layout()
    plt.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.92, wspace=0.01, hspace=0.01)
    # Plot subplots
    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            ax[i][j].axis('off')
            ax[i][j].autoscale(True)
            # Display base images
            if j == 0:    # Original image display
                ax[i][j].imshow(im)
                title = 'Original'
            elif j == 1:    # Heatmap label display
                ax[i][j].imshow(hm)
                title = 'Heatmap'
            else:    # Original + heatmap overlay display
                ax[i][j].imshow(im_hm)
                title = 'Original with Heatmap'
            # Display predicted labels
            if i == 0:    # RAW
                ax[i][j].set_title(title)
            elif i == 1:    # Overall maximum prediction display
                if len(coord_abs) == 0:
                    ax[i][j].text(10, 10, 'No prediction!', ha='left', va='top', bbox=dict(ec='r', fc='r', alpha=0.5))
                else:
                    ax[i][j].plot(coord_abs[1], coord_abs[0], 'r.', ms=2.5)
                ax[i][j].set_title('Overall Max - ' + title)
            else:    # Local maxima prediction display
                if coord_loc.shape[0] == 0:
                    ax[i][j].text(10, 10, 'No prediction!', ha='left', va='top', bbox=dict(ec='r', fc='r', alpha=0.5))
                else:
                    ax[i][j].plot(coord_loc[:, 1], coord_loc[:, 0], 'r.', ms=2.5)
                ax[i][j].set_title('Local Max - ' + title)
    # Display results
    plt.show()


def btm_mirror_paw_fix(btm_hm_pred, sd_hm_pred):
    """ Correct animal paw switch issue for Bottom-Mirror type setup.

    Args:
        btm_hm_pred (list[dict]): List of dictionary with bottom-view HeatMap label info.
        sd_hm_pred (list[dict]): List of dictionary with side-view HeatMap label info.

    Returns:
        tuple[list[dict], list[dict]]:
            btm_lbl (list[dict]): List of dictionary with JSON type bottom-view label info.
            sd_lbl (list[dict]): List of dictionary with JSON type side-view label info.
    """
    # Unravel predictions
    btm_abs, btm_loc = hmpred_unravel(btm_hm_pred)
    sd_abs, sd_loc = hmpred_unravel(sd_hm_pred)
    # Compute basic info
    flag = True if (btm_abs['Snout'][1] - btm_abs['Tail 01'][1]) > 0 else False  # Detect walking direction, True = LEFT
    lbl_temp = {'left': -1, 'top': -1, 'width': 0, 'height': 0, 'label': 'NULL'}    # INIT VAR

    # Get and correct bottom-view label
    btm_lbl = []    # INIT VAR
    for tag in btm_abs:
        if ('Paw' in tag) and (btm_loc[tag].shape[0] > 1):
            if flag:
                if 'Left' in tag:
                    sel = btm_loc[tag][np.argmin(btm_loc[tag], axis=1)[0], :]    # Heading right, left-paws higher
                else:
                    sel = btm_loc[tag][np.argmax(btm_loc[tag], axis=1)[0], :]    # Heading right, right-paws lower
            else:
                if 'Left' in tag:
                    sel = btm_loc[tag][np.argmax(btm_loc[tag], axis=1)[0], :]    # Heading left, left-paws lower
                else:
                    sel = btm_loc[tag][np.argmin(btm_loc[tag], axis=1)[0], :]    # Heading left, right-paws higher
            lbl_temp['left'] = sel[1].item()
            lbl_temp['top'] = sel[0].item()
            lbl_temp['width'] = lbl_temp['height'] = 1
            lbl_temp['label'] = tag
        elif len(btm_abs[tag]) != 0:
            lbl_temp['left'] = btm_abs[tag][1].item()
            lbl_temp['top'] = btm_abs[tag][0].item()
            lbl_temp['width'] = lbl_temp['height'] = 1
            lbl_temp['label'] = tag
        else:
            lbl_temp = {'left': -1, 'top': -1, 'width': 0, 'height': 0, 'label': tag}
        btm_lbl.append(copy.deepcopy(lbl_temp))

    # Passing bottom-view info to side-view (shared horizontal axis)
    btm_sd_info = {}    # INIT VAR
    for lbl in btm_lbl:
        if 'Paw' in lbl['label']:
            btm_sd_info[lbl['label']] = lbl['left']
    # Get and correct side-view label
    sd_lbl = []    # INIT VAR
    for tag in sd_abs:
        if ('Paw' in tag) and (sd_loc[tag].shape[0] > 1):
            sel = sd_loc[tag][np.argmin(abs(sd_loc[tag][:, 1] - btm_sd_info[tag])), :]
            lbl_temp['left'] = sel[1].item()
            lbl_temp['top'] = sel[0].item()
            lbl_temp['width'] = lbl_temp['height'] = 1
            lbl_temp['label'] = tag
        elif len(sd_abs[tag]) != 0:
            lbl_temp['left'] = sd_abs[tag][1].item()
            lbl_temp['top'] = sd_abs[tag][0].item()
            lbl_temp['width'] = lbl_temp['height'] = 1
            lbl_temp['label'] = tag
        else:
            lbl_temp = {'left': -1, 'top': -1, 'width': 0, 'height': 0, 'label': tag}
        sd_lbl.append(copy.deepcopy(lbl_temp))

    return btm_lbl, sd_lbl
