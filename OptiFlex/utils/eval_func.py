import os
import copy
import math
import numpy as np
from OptiFlex.utils.base_func import prog_print
from OptiFlex.data.lbl_proc import jsl_read, hml_read

"""Function list:
get_ref_lbl(lbl_data, bb_meas):  Calculate a reference label data from original label data.
predict_eval(prd_lbl, ref_lbl, mode):  Evaluate predicted label data quality with defined method.
heatmap_eval(js_prd, hm_ref):  Evaluate prediction accuracy of HeatMap (converted to JSON).
eval_pck(js_data, hm_data):  Evaluation metric PCK.
grp_eval_pck(js_dir, hm_dir, lbl_lst):  Group evaluation metric PCK.
eval_hmclose(hm_prd, hm_ref, tol):  Evaluate if predicted HeatMap is close enough to reference HeatMap.
grp_eval_hmclose(hm_prd_dir, hm_ref_dir, lbl_lst, tol):  Group evaluation metric HMClose.
"""


def get_ref_lbl(lbl_data, bb_meas=None):
    """ Calculate a reference label date from original label data.

    Args:
        lbl_data (list[dict]): List of dictionary with label info.
        bb_meas (dict[str, tuple[int, int]] or None): Define bounding box size for each type of labels (default: None).
                --  KEY (str) = label name
                --  VALUE (tuple[int, int]) = (width, height)

    Returns:
        list[dict]: List of dictionary with reference label info to be used for evaluation.
    """
    lbl_temp = {'x': None, 'y': None, 'w': None, 'h': None, 'label': None}    # INIT VAR
    ref_lbl = []    # INIT VAR

    for lbl in lbl_data:
        # Get center value of original label
        lbl_temp['x'] = lbl['left'] + lbl['width'] / 2
        lbl_temp['y'] = lbl['top'] + lbl['height'] / 2
        # Set bounding box for reference
        if bb_meas is None:
            lbl_temp['w'] = lbl['width']
            lbl_temp['h'] = lbl['height']
        else:
            lbl_temp['w'] = bb_meas[lbl['label']][0]
            lbl_temp['h'] = bb_meas[lbl['label']][1]
        # Get original label name
        lbl_temp['label'] = lbl['label']
        ref_lbl.append(copy.deepcopy(lbl_temp))

    return ref_lbl


def predict_eval(prd_lbl, ref_lbl, mode=0):
    """ Evaluate predicted label data quality with defined method.

    Args:
        prd_lbl (list[dict]): List of dictionary with predicted label info.
        ref_lbl (list[dict]): List of dictionary with reference label info to be used for evaluation.
        mode (int): Evaluation method to be used for qualification:
                --  0 = Distance between PREDICTED and REFERENCE center (ratio)
                --  1 = Distance between PREDICTED and REFERENCE center (absolute)
                --  2 = Area of intersection of PREDICTED and REFERENCE bounding box (ratio)
                --  3 = Area of intersection of PREDICTED and REFERENCE bounding box (absolute)
                --  4 = Mean absolute error (MAE)

    Returns:
        dict[str, float]: Dictionary with KEY of label name, VALUE of quality.
    """
    prd_copy = copy.deepcopy(prd_lbl)    # Make copy of original data, avoid unexpected modification
    ref_copy = copy.deepcopy(ref_lbl)    # Make copy of original data, avoid unexpected modification
    qual_eval = dict()    # INIT VAR
    for ref_data in ref_copy:
        found = False    # Reset flag each loop
        for prd_data in prd_copy:
            if prd_data['label'] == ref_data['label']:
                found = True    # Set flag when found
                qual = 0    # INIT VAR

                # Prediction quality evaluation
                if mode == 0:    # MODE 0: distance ratio
                    prd_x = prd_data['left'] + prd_data['width'] / 2
                    # All the corresponding sides have lengths in the same ratio in similar triangles
                    qual = 2 * abs(ref_data['x'] - prd_x) / ref_data['width']
                elif mode == 1:    # MODE 1: distance absolute
                    prd_x = prd_data['left'] + prd_data['width'] / 2
                    prd_y = prd_data['top'] + prd_data['height'] / 2
                    qual = math.hypot((ref_data['x'] - prd_x), (ref_data['y'] - prd_y))
                elif (mode == 2) or (mode == 3):    # MODE 2 or 3: area modes
                    # Get rectangle coordinates of reference data
                    ref_x1 = ref_data['x'] - ref_data['w'] / 2
                    ref_x2 = ref_data['x'] + ref_data['w'] / 2
                    ref_y1 = ref_data['y'] - ref_data['h'] / 2
                    ref_y2 = ref_data['y'] + ref_data['h'] / 2
                    # Get rectangle coordinates of predicted data
                    prd_x1 = prd_data['left']
                    prd_x2 = prd_data['left'] + prd_data['width']
                    prd_y1 = prd_data['top']
                    prd_y2 = prd_data['top'] + prd_data['height']
                    # Calculate the area of the intersection, and its ratio
                    area_intsec = (max(0, min(ref_x2, prd_x2) - max(ref_x1, prd_x1)) *
                                   max(0, min(ref_y2, prd_y2) - max(ref_y1, prd_y1)))
                    if mode == 2:    # MODE 2: intersect area ratio
                        area_tot = ref_data['w'] * ref_data['h'] + prd_data['width'] * prd_data['height'] - area_intsec
                        qual = area_intsec / area_tot
                    elif mode == 3:    # MODE 3: intersect area absolute
                        qual = area_intsec
                elif mode == 4:
                    prd_x = prd_data['left'] + prd_data['width'] / 2
                    prd_y = prd_data['top'] + prd_data['height'] / 2
                    qual = (abs(ref_data['x'] - prd_x) + abs(ref_data['y'] - prd_y)) / 2
                else:
                    print('Evaluation mode:[%d] not defined, function out!' % mode)
                    exit()

                qual_eval[ref_data['label']] = qual
                ref_copy.remove(ref_data)    # Avoid getting same label multiple times
                prd_copy.remove(prd_data)    # Avoid getting same label multiple times
                break    # Avoid pass-over when found

        if not found:
            qual_eval[ref_data['label']] = None
            print('Label [%s] missing in predicted data!' % ref_data['label'])
            ref_copy.remove(ref_data)    # Avoid getting same label multiple times

    return qual_eval


def heatmap_eval(js_prd, hm_ref):
    """ Evaluate prediction accuracy of HeatMap (converted to JSON).

    Args:
        js_prd (list[dict]): List of dictionary with predicted JSON type label info.
        hm_ref (list[dict]): List of dictionary with reference HeatMap type label info.

    Returns:
        list[dict]: Evaluation results.
                --  KEY_'val' (float): HeatMap readout from JSON.
                --  KEY_'tgt' (int): {0 or 1} If the readout larger than threshold.
                --  KEY_'label' (str): Label name.

    """
    hm_dat = copy.deepcopy(hm_ref)    # Make copy of original data, avoid unexpected modification
    eval_data = {'val': None, 'tgt': None, 'label': None}    # INIT VAR
    eval_list = []    # INIT VAR
    for lbl in js_prd:
        for hm in hm_dat:
            if lbl['label'] == hm['label']:
                heatmap = hm['heatmap'] * (1 / hm['heatmap'].max())    # Normalize data
                eval_data['label'] = lbl['label']
                eval_data['val'] = heatmap[lbl['top'], lbl['left']].item()
                eval_data['tgt'] = 1 if eval_data['val'] > 0 else 0
                hm_dat.remove(hm)    # Avoid getting same label multiple times
                eval_list.append(copy.deepcopy(eval_data))
    return eval_list


def eval_pck(js_data, hm_data):
    """ Evaluation metric PCK.

    Args:
        js_data (list[dict]): List of dictionary with predicted JSON type label info.
        hm_data (list[dict]): List of dictionary with reference HeatMap GT label info.

    Returns:
        dict[str, tuple[float, int]]:
            --  KEY (str): label name.
            --  VALUE (tuple[float, int]) = (value, quality):
                --  value (float): HeatMap value of JSON coordinates.
                --  quality (int): if value > 0, defines as accurate, quality = 1, else quality = 0.
    """
    js_copy = copy.deepcopy(js_data)    # Make a safe copy
    res = {}    # INIT VAR
    # Compute evaluation
    for hm in hm_data:
        flag = True    # INIT/RESET VAR: No predicted label not found
        for js in js_copy:
            if hm['label'] == js['label']:
                flag = False    # Predicted label found
                if hm['heatmap'].max() == 0 or hm['heatmap'] is None:    # Detect HM empty label situation
                    if js['top'] is None and js['left'] is None:    # JS label also empty, mark as accurate
                        val = 1.0
                        qal = 1
                    else:    # JS label is not empty, mark as fault
                        val = 0.0
                        qal = 0
                else:
                    if js['top'] is None and js['left'] is None:    # JS label also empty, mark as accurate
                        val = 0.0
                        qal = 0
                    else:
                        val = hm['heatmap'][js['top'], js['left']].item() / hm['heatmap'].max().item()
                        qal = 1 if val > 0 else 0    # Detect if prediction hit on heatmap
                res[hm['label']] = (val, qal)
                js_copy.remove(js)    # Avoid loop over
        if flag:
            res[hm['label']] = (.0, 0)    # Handel missing prediction
    return res


def grp_eval_pck(js_dir, hm_dir, lbl_lst):
    """ Group evaluation metric PCK.

    Args:
        js_dir (str): Directory containing predicted JSON labels.
        hm_dir (str): Directory containing reference HeatMap labels.
        lbl_lst (list[str]): Joint names in label be evaluated.

    Returns:
        dict[str, tuple[tuple[float, float], tuple[float, float]]]:
            --  KEY (str): label name.
            --  VALUE (tuple[tuple[float, float], tuple[float, float]]) = (stat of value, stat of quality):
                --  stat of value (tuple[float, float]): (mean, standard error)
                --  stat of quality (tuple[float, float]): (mean, standard error)
    """
    null_res = {v: (.0, 0) for v in lbl_lst}    # SET DUMMY
    hm_lst = []    # INIT VAR
    js_lst = []    # INIT VAR
    # Get all required files
    for f in os.listdir(hm_dir):
        if f.endswith('.pkl'):
            hm_lst.append(os.path.join(hm_dir, f))
            js_lst.append(os.path.join(js_dir, (os.path.splitext(f)[0] + '.json')))
    # Compute group evaluation
    n = len(hm_lst)
    fin_res = {v: np.empty((2, n), dtype=float) for v in lbl_lst}    # INIT VAR
    for i in range(n):
        hm_in = hml_read(hm_lst[i])
        if os.path.isfile(js_lst[i]):
            js_in = jsl_read(js_lst[i])
            sgl_res = eval_pck(js_in, hm_in)
        else:
            sgl_res = null_res
        for lbl in lbl_lst:
            fin_res[lbl][:, i] = sgl_res[lbl]
        prog_print(i + 1, n, "PCK evaluation processed:")
    # Get statistics
    stat_res = {}    # INIT VAR
    for lbl in lbl_lst:
        avg_val, avg_qal = np.mean(fin_res[lbl], axis=1)
        std_val, std_qal = np.std(fin_res[lbl], axis=1)
        stat_res[lbl] = ((avg_val, std_val), (avg_qal, std_qal))
    return stat_res


def pck_model_comp(model_prd_dir, hm_ref_dir, lbl_lst, mode=1):
    """ Compare models with PCK evaluation method.

    Args:
        model_prd_dir (dict[str, str]): Model prediction results directories.
                --  KEY (str) = model name
                --  VALUE (str) = model prediction directory
        hm_ref_dir (str): Reference (GT) HeatMap label directory.
        lbl_lst (list[str]): List of joint names to be compared.
        mode (int): Model compare method (default: 1).
                --  0 = SCORE - average reference HeatMap readout value
                --  1 = ACCURACY - average reference HeatMap on-target value

    Returns:
        tuple[dict[str, dict[str, tuple[tuple[float, float], tuple[float, float]]]], list[str]]:
            res_dic (dict[str, dict[str, tuple[tuple[float, float], tuple[float, float]]]]): Results.
                --  KEY (str): label name.
                --  VALUE (tuple[tuple[float, float], tuple[float, float]]) = (stat of value, stat of quality):
                    --  stat of value (tuple[float, float]): (mean, standard error)
                    --  stat of quality (tuple[float, float]): (mean, standard error)
            best (list[str]): Best model(s) name(s).
    """
    # Evaluation
    model_lst = []    # INIT VAR
    res_dic = {}    # INIT VAR
    for model in model_prd_dir:
        print("Processing predicted labels from model [%s]..." % model)
        model_lst.append(model)
        res_temp = grp_eval_pck(model_prd_dir[model], hm_ref_dir, lbl_lst)
        res_dic[model] = res_temp
    print()

    # Statistics and report
    repo_str = "%%-0%ds    %%s%%s" % max(len(max(lbl_lst, key=len)), 10)    # INIT VAR, 10 = len("Joint_Name")
    # Print header
    dat_str = str()    # INIT VAR
    spacer = []    # INIT VAR
    count = []    # INIT VAR
    for ms in model_lst:
        spacer.append(max(len(ms), 8))    # 8 = len(%8.6f))
        dat_str_base = "%%0%ds    " % max(len(ms), 8)    # 8 = len(%8.6f))
        dat_str += dat_str_base % ms
        count.append(0)
    print(repo_str % ("Joint_Name", dat_str, "Best_Model(s)"))
    # Get and print result statistics
    for lbl in lbl_lst:
        dat_str = str()    # RESET VAR
        cmp_lst = []    # INIT VAR
        for i in range(len(res_dic)):
            cmp_lst.append(res_dic[model_lst[i]][lbl][mode][0])
            dat_str_base = "%%%d.6f    " % spacer[i]
            dat_str += dat_str_base % res_dic[model_lst[i]][lbl][mode][0]
        win = [i for i, j in enumerate(cmp_lst) if j == max(cmp_lst)]
        model_str = str()    # INIT VAR
        for i in win:
            model_str += "[" + model_lst[i] + "] "
            count[i] += 1
        print(repo_str % (lbl, dat_str, model_str.rstrip()))

    # Conclusion
    best_idx = [i for i, j in enumerate(count) if j == max(count)]
    best = []    # INIT VAR
    best_str = str()    # INIT VAR
    for i in best_idx:
        best.append(model_lst[i])
        best_str += "[" + model_lst[i] + "] "
    print("Best model overall is %s" % best_str.rstrip())
    return res_dic, best


def eval_hmclose(hm_prd, hm_ref, tol=0.5):
    """ Evaluate if predicted HeatMap is close enough to reference HeatMap.

    Args:
        hm_prd (list[dict]): List of dictionary with predicted HeatMap label info.
        hm_ref (list[dict]): List of dictionary with reference HeatMap GT label info.
        tol (float): Relative tolerance, defined as [abs(hm_ref) * tol] (default = 0.5).

    Returns:
        dict[str, int]:
            --  KEY (str): label name.
            --  VALUE (float) = Mean of closed value.
    """
    res = {}  # INIT VAR
    for lref in hm_ref:
        flag = True
        for lprd in hm_prd:
            if lref["label"] == lprd["label"]:
                if lprd['heatmap'].max() > 0:
                    flag = False
                    mref = np.multiply(lref['heatmap'], 1 / lref['heatmap'].max())    # Normalize
                    mprd = np.multiply(lprd['heatmap'], 1 / lprd['heatmap'].max())    # Normalize
                    comp = np.isclose(mprd, mref, atol=0.0, rtol=tol).astype(int)    # [atol = 0.0] ignore 0 values
                    res[lref["label"]] = np.mean(comp)
        if flag:
            res[lref["label"]] = 0
    return res


def grp_eval_hmclose(hm_prd_dir, hm_ref_dir, lbl_lst, tol=0.5):
    """ Group evaluation metric HMClose.

    Args:
        hm_prd_dir (str): Directory containing predicted HeatMap labels.
        hm_ref_dir (str): Directory containing reference HeatMap labels.
        lbl_lst (list[str]): Joint names in label be evaluated.
        tol (float): Relative tolerance, defined as [abs(hm_ref) * tol] (default = 0.5).

    Returns:
        dict[str, tuple[float, float]]]:
            --  KEY (str): label name.
            --  VALUE (tuple[tuple[float, float]): Stat of value (mean, standard error)
    """
    ref_lst = [os.path.join(hm_ref_dir, f) for f in os.listdir(hm_ref_dir) if f.endswith(".pkl")]
    prd_lst = [os.path.join(hm_prd_dir, f) for f in os.listdir(hm_ref_dir) if f.endswith(".pkl")]
    null_res = {v: 0 for v in lbl_lst}    # SET DUMMY
    n = len(ref_lst)
    fin_res = {v: np.empty(n, dtype=float) for v in lbl_lst}    # INIT VAR
    for i in range(n):
        hm_refa = hml_read(ref_lst[i])
        if os.path.isfile(prd_lst[i]):
            hm_prda = hml_read(prd_lst[i])
            sgl_res = eval_hmclose(hm_refa, hm_prda, tol)
        else:
            sgl_res = null_res
        for lbl in lbl_lst:
            fin_res[lbl][i] = sgl_res[lbl]
        prog_print(i + 1, n, "HMClose evaluation progress:")
    # Get statistics
    stat_res = {}    # INIT VAR
    for lbl in lbl_lst:
        avg_qal = np.mean(fin_res[lbl], axis=None)
        std_qal = np.std(fin_res[lbl], axis=None)
        stat_res[lbl] = (avg_qal, std_qal)
    return stat_res
