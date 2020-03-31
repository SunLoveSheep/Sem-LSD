from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
# /workspace/tangyang.sy/pytorch_CV/pytorch_CenterNet/src

try:
    from utils_ctdet.lineNMS import do_nms_line, do_acl_line, do_acl_line_v1
except ImportError:
    from utils.lineNMS import do_nms_line, do_acl_line, do_acl_line_v1

from collections import OrderedDict
import numpy as np
import os
import xml.etree.ElementTree as ET

LABEL_MAP_KAIST = [
    'building', 'ground_building', 'wall', 'ground_wall',
    'grass', 'fence', 'ground_fence', 'pole', 'curb', 'sign', 'tree', 'window', 'door', 'bridge'
]

LABEL2SKIP = [
    'sign', 'window', 'tree',
]

nms_func = {
    'acl': do_acl_line,
    'acl_v1': do_acl_line_v1,
    'iou': do_nms_line,
}

METRIC = 'acl'
METRIC_EVAL = 'acl'
IF_CORR_FN = True
METRIC_THRESH = 0.9
DEFAULT_DATASET = 'OBJLINE_KAIST'

# mAP metrics:
IF_mAP = True
mAP_score = 0.5
LV_IOU = [0.5, 0.75, 0.9] if not IF_mAP else [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
# LV_IOU = [0.5, 0.75, 0.9] if not IF_mAP else [0.5, 0.9]
mAP_SUB = ['building', 'pole', 'curb']

LV_ACL = LV_IOU
LV_SCORE = [0.25, 0.5, 0.6, 0.7, 0.8, 0.9]
ROC_score = 0.25
if mAP_score not in LV_SCORE:
    LV_SCORE.append(mAP_score)

root = '/workspace/tangyang.sy/pytorch_CV/test_imgs/KAIST_5seqs_20200214/'

MODEL = 'best'
# EXP_ID = 'kaist_5seqs_ResNet18_GradMagLoss_NegOnly_20200224_ep{}_vis0.25'.format(MODEL)
EXP_ID = 'kaist_5seqs_ResNet18_20200224_ep{}_vis0.25'.format(MODEL)

SRC_GT = root + 'Annotations/'

DATASET_MAP = {
    'KAIST': LABEL_MAP_KAIST,
}

DATASET = 'KAIST'
LABEL_MAP = DATASET_MAP[DATASET]
LABEL_MAP.append('lineseg')  # For LSD and AFM results.

SRC_PRED = root + 'Preds_{}/'.format(EXP_ID)
PTH_EXT = '' if not IF_CORR_FN else '_corrFN'
RECORD_PTH = root + 'summary{}/'.format(PTH_EXT)
if not os.path.exists(RECORD_PTH):
    os.makedirs(RECORD_PTH)
if len(LABEL2SKIP) == 0:
    RECORD_TXT = RECORD_PTH + 'res_id-{}_metric-{}-{}.txt'.format(EXP_ID, METRIC, METRIC_THRESH)
    RECORD_perCate_TXT = RECORD_PTH + 'res_perCate_id-{}_metric-{}-{}.txt'.format(EXP_ID, METRIC, METRIC_THRESH)
else:
    RECORD_TXT = RECORD_PTH + 'res_id-{}_metric-{}-{}-skip{}.txt'.format(EXP_ID, METRIC, METRIC_THRESH, len(LABEL2SKIP))
    RECORD_perCate_TXT = RECORD_PTH + 'res_perCate_id-{}_metric-{}-{}-skip{}.txt'.format(EXP_ID, METRIC, METRIC_THRESH,
                                                                                         len(LABEL2SKIP))

lst_gt = [x for x in sorted(os.listdir(SRC_GT)) if '.xml' in x]
lst_pred = [x for x in sorted(os.listdir(SRC_PRED)) if '.xml' in x]

PI = 3.14159265359


class LineSeg:
    x_left = 0  # x_left
    y_left = 0  # y_left
    x_right = 0  # x_right
    y_right = 0  # y_right
    cate = 'null'
    score = 0.0
    direction = None
    angle = 0.0  # angle against positive x-axis
    length = 0.0

    def __init__(self, x_left_in, y_left_in, x_right_in, y_right_in,
                 cate_in, score_in, direct_in):
        self.x_left = x_left_in
        self.y_left = y_left_in
        self.x_right = x_right_in
        self.y_right = y_right_in
        self.cate = cate_in
        self.score = score_in
        self.direction = direct_in
        self.angle = self._cal_ang()

    def _cal_ang(self):
        ang_res = 0.
        return ang_res


def _fast_reject(line1, line2):
    # Fast reject:
    if line1.x_right < line2.x_left or line1.x_left > line2.x_right:
        return True
    line1_ymin = min(line1.y_left, line1.y_right)
    line1_ymax = max(line1.y_left, line1.y_right)
    line2_ymin = min(line2.y_left, line2.y_right)
    line2_ymax = max(line2.y_left, line2.y_right)
    if line1_ymin > line2_ymax or line1_ymax < line2_ymin:
        return True
    return False


def _cal_iou_line(line1, line2):
    # if _fast_reject(line1, line2):
    #    return 0.

    # if line1.direction != line2.direction:
    #     return 0.

    line1_xmin = min(line1.x_left, line1.x_right)
    line1_ymin = min(line1.y_left, line1.y_right)
    line1_xmax = max(line1.x_left, line1.x_right)
    line1_ymax = max(line1.y_left, line1.y_right)

    line2_xmin = min(line2.x_left, line2.x_right)
    line2_ymin = min(line2.y_left, line2.y_right)
    line2_xmax = max(line2.x_left, line2.x_right)
    line2_ymax = max(line2.y_left, line2.y_right)

    inter_xmin = max(line1_xmin, line2_xmin)
    inter_ymin = max(line1_ymin, line2_ymin)
    inter_xmax = min(line1_xmax, line2_xmax)
    inter_ymax = min(line1_ymax, line2_ymax)
    inter_x = max(0, inter_xmax - inter_xmin)
    inter_y = max(0, inter_ymax - inter_ymin)
    area_inter = inter_x * inter_y

    union_xmin = min(line1_xmin, line2_xmin)
    union_ymin = min(line1_ymin, line2_ymin)
    union_xmax = max(line1_xmax, line2_xmax)
    union_ymax = max(line1_ymax, line2_ymax)

    union_x = 1. if union_xmax == union_xmin else union_xmax - union_xmin
    union_y = 1. if union_ymax == union_ymin else union_ymax - union_ymin
    area_union = union_x * union_y

    iou = area_inter / area_union

    return iou


# line1 should be ground truth, if available
def _cal_acl_line(line1, line2):
    # if _fast_reject(line1, line2):
    #    return 0.
    # if line1.direction != line2.direction:
    #    return 0.

    sum1_x = line1.x_left + line1.x_right
    sum1_y = line1.y_left + line1.y_right
    c1_x = sum1_x / 2
    c1_y = sum1_y / 2
    l1_wr = np.sqrt(sum1_x * sum1_x + sum1_y * sum1_y)
    l1_x = line1.x_right - line1.x_left
    l1_y = line1.y_right - line1.y_left
    l1 = np.sqrt(l1_x * l1_x + l1_y * l1_y)
    alpha1 = (line1.y_right - line1.y_left) / (line1.x_right - line1.x_left) if \
        line1.x_right != line1.x_left else (line1.y_right - line1.y_left) / 1.0
    alpha1 = np.abs(np.arctan(alpha1) * 180 / PI)

    sum2_x = line2.x_left + line2.x_right
    sum2_y = line2.y_left + line2.y_right
    c2_x = sum2_x / 2
    c2_y = sum2_y / 2
    l2_wr = np.sqrt(sum2_x * sum2_x + sum2_y * sum2_y)
    l2_x = line2.x_right - line2.x_left
    l2_y = line2.y_right - line2.y_left
    l2 = np.sqrt(l2_x * l2_x + l2_y * l2_y)
    alpha2 = (line2.y_right - line2.y_left) / (line2.x_right - line2.x_left) if \
        line2.x_right != line2.x_left else (line2.y_right - line2.y_left) / 1.0
    alpha2 = np.abs(np.arctan(alpha2) * 180 / PI)

    sim_a = max(0, 1 - (np.abs(alpha1 - alpha2) * 0.0111111111111))
    sim_c = max(0, 1 - (np.sqrt((c2_x - c1_x) * (c2_x - c1_x) + (c2_y - c1_y) * (c2_y - c1_y)))
                / (l1 * 0.5))
    sim_l = max(0, 1 - np.abs(l1 - l2) / l1)
    sim_l_wr = max(0, 1 - np.abs(l1_wr - l2_wr) / l1_wr)
    #print("sim l: {:.3f} | sim l wr: {:.3f} | line1: {},{},{},{} | line2: {},{},{},{}".format(
    #    sim_l, sim_l_wr,
    #    line1.x_left, line1.y_left, line1.x_right, line1.y_right,
    #    line2.x_left, line2.y_left, line1.x_right, line2.y_right
    #))
    acl = sim_a * sim_c * sim_l

    return acl


def _gaussian_radius(height, width, min_overlap=0.7):
    # a3 = 4 * min_overlap
    # b3 = -2 * min_overlap * (height + width)
    # c3 = (min_overlap - 1) * width * height
    # sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    # r3 = (b3 + sq3) / 2
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2
    return r1


def _cal_acl_line_v1(line1, line2):
    # if _fast_reject(line1, line2):
    #    return 0.

    # Vectorize to calculate cosine angle
    v_x1 = line1.x_right - line1.x_left
    v_y1 = line1.y_right - line1.y_left
    v_x2 = line2.x_right - line2.x_left
    v_y2 = line2.y_right - line2.y_left
    l1 = np.sqrt(v_x1 * v_x1 + v_y1 * v_y1)
    l2 = np.sqrt(v_x2 * v_x2 + v_y2 * v_y2)
    cos_a = (v_x1 * v_x2 + v_y1 * v_y2) / (l1 * l2)

    # Gaussian distribution to get score of center point
    radius = _gaussian_radius(v_y1, v_x1)
    sigma = (2 * radius - 1)/6
    sum1_x = line1.x_left + line1.x_right
    sum1_y = line1.y_left + line1.y_right
    c1_x = sum1_x / 2
    c1_y = sum1_y / 2
    sum2_x = line2.x_left + line2.x_right
    sum2_y = line2.y_left + line2.y_right
    c2_x = sum2_x / 2
    c2_y = sum2_y / 2
    d_x = c2_x - c1_x
    d_y = c2_y - c1_y
    c_score = np.exp(-(d_x * d_x + d_y * d_y) / (2 * sigma * sigma))

    sim_a = cos_a
    sim_c = c_score
    sim_l = max(0, 1 - np.abs(l1 - l2) / l1)

    return sim_a * sim_c * sim_l


metric_func = {
    'acl': _cal_acl_line,
    'acl_v1': _cal_acl_line_v1,
    'iou': _cal_iou_line,
}


def reverse_direct(lines):
    lines_out = lines.copy()
    for line in lines_out:
        line.direction = 1 - line.direction
        y_tmp = line.y_left
        line.y_left = line.y_right
        line.y_right = y_tmp
    return lines_out


def get_lines_from_xml(xml_file, if_gt=False):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    lst_lines = list()
    for obj in root.findall('object'):
        objline = obj.find('bndbox')

        name = obj.find('name').text
        if name is None:
            continue
        name = name.lower()
        score = obj.find('score').text if not if_gt else 1.0
        try:
            direct_str = obj.find('direction').text
            direction = 1.0 if direct_str == 'lt2rb' else 0.0
        except AttributeError:
            direction = 2.0

        # The followings are actually already converted to x_left, y_left, x_right, and y_right
        xmin = int(float(objline.find('xmin').text))
        ymin = int(float(objline.find('ymin').text))
        xmax = int(float(objline.find('xmax').text))
        ymax = int(float(objline.find('ymax').text))
        if direction == 1.0:
            x_left = min(xmin, xmax)
            y_left = min(ymin, ymax)
            x_right = max(xmin, xmax)
            y_right = max(ymin, ymax)
        elif direction == 0.0:
            x_left = min(xmin, xmax)
            y_left = max(ymin, ymax)
            x_right = max(xmin, xmax)
            y_right = min(ymin, ymax)
        else:  # direction == 2.0, bbox
            x_left = xmin
            y_left = ymin
            x_right = xmax
            y_right = ymax

        # if not if_gt:
        #    direction = obj.find('direction').text
        # else:
        #    direct_str = 'lb2rt' if x_left < x_right and y_left < y_right else 'lt2rb'
        #    direction = 1.0 if direct_str == 'lt2rb' else 0.0

        # line = [xmin, ymin, xmax, ymax, name, score, direction]
        line = LineSeg(
            x_left_in=int(x_left),
            y_left_in=int(y_left),
            x_right_in=int(x_right),
            y_right_in=int(y_right),
            cate_in=name,
            score_in=float(score),
            direct_in=float(direction)
        )
        lst_lines.append(line)

    return lst_lines


# given ground truth objects and predicted objects.
# given 3M/5M mask and levels of IoU and confidence to test
# return resultant number of precision / recall objects
# TP (True Positive): predict the box with correct category
# TN (True Negative): predict the box with wrong category
# FP (False Positive): predict a box when there is no box
# FN (False Negative): predict no box when there is a box
# Output dictionary format:
# IoU_thres_1 -
#   Score_thres_1 - TP, FP, FN, TN
#   Score_thres_2 - TP, FP, FN, TN
#   ...
#   Score_thres_N - TP, FP, FN, TN
# IoU_thres_2 -
#   Score_thres_1 - TP, FP, FN, TN
#   Score_thres_2 - TP, FP, FN, TN
#   ...
#   Score_thres_N - TP, FP, FN, TN
# ...
# IoU_thres_M -
#   Score_thres_1 - TP, FP, FN, TN
#   Score_thres_2 - TP, FP, FN, TN
#   ...
#   Score_thres_N - TP, FP, FN, TN
def cal_metric_res(
        lines_gt_in,
        lines_pred_in,
        metric='acl'
):
    # initiate output dict
    dict_res = OrderedDict((key_iou, 0) for key_iou in LV_IOU)
    for key_iou in dict_res.keys():
        dict_res[key_iou] = OrderedDict((key_score, 0) for key_score in LV_SCORE)
        for key_score in dict_res[key_iou].keys():
            dict_res[key_iou][key_score] = {
                'TP': 0,
                'TN': 0,
                'FP': 0,
                'FN': 0,
            }

    # initiate output dict with Cate level:
    dict_res_perCate = OrderedDict((key, 0) for key in LABEL_MAP if key not in LABEL2SKIP)
    for key_cate in dict_res_perCate.keys():
        dict_res_perCate[key_cate] = OrderedDict((key_iou, 0) for key_iou in LV_IOU)
        for key_iou in dict_res_perCate[key_cate].keys():
            dict_res_perCate[key_cate][key_iou] = OrderedDict((key_score, 0) for key_score in LV_SCORE)
            for key_score in dict_res_perCate[key_cate][key_iou].keys():
                dict_res_perCate[key_cate][key_iou][key_score] = {
                    'TP': 0,
                    'TN': 0,
                    'FP': 0,
                    'FN': 0,
                }

    # ------ loop through ground truth bboxes to find FN (missed bboxes) ------
    for line in lines_gt_in:
        # Each gt should be counted as TP or FN, this is to record if it has been recorded
        if_recorded = 0

        # Reivse some typo in prediction cates
        line.cate = 'ground_building' if line.cate == 'ground building' else line.cate

        if line.cate in LABEL2SKIP or line.cate not in LABEL_MAP:
            # print("Found invalid line type: {}, skip.".format(line.cate))
            continue
        # compare bbox with all ground truth bboxes
        # to find the one with largest IoU:
        max_iou = 0
        if_cate_right = False
        score_at_max_iou = 0.
        for line_pred in lines_pred_in:
            # modified to support no-semantic line segments from LSD:
            if (line_pred.cate not in LABEL_MAP and line_pred.cate != 'lineseg')\
                    or (line_pred.cate in LABEL2SKIP):
                continue

            if line.cate in LABEL_MAP_OBJ_KAIST and line_pred.cate in LABEL_MAP_OBJ_KAIST:
                iou = _cal_iou_line(line, line_pred)
            else:
                iou = metric_func[metric](line, line_pred)
                # print(iou)
                # iou = _cal_acl_line(line, line_pred) if metric == 'acl' else _cal_iou_line(line, line_pred)
                # iou_test = _cal_acl_line_v1(line, line_pred)
                # print(iou_test)
            if iou > max_iou:
                max_iou = iou
                score_at_max_iou = line_pred.score
                if line.cate == line_pred.cate:
                    if_cate_right = True
                else:
                    if_cate_right = False if line_pred.cate != 'lineseg' else True

        # arrange output according to levels of IoU and levels of score to check:
        for lv_iou in LV_IOU:
            # if IoU larger than current IoU threshold:
            if max_iou > lv_iou:
                for lv_score in LV_SCORE:
                    # if score larger than current score threshold:
                    if score_at_max_iou > lv_score:
                        if if_cate_right:
                            dict_res[lv_iou][lv_score]['TP'] += 1
                            dict_res_perCate[line.cate][lv_iou][lv_score]['TP'] += 1
                        else:
                            # FP increase by 1, since this pred has no gt (diff cate)
                            dict_res[lv_iou][lv_score]['FP'] += 1
                            dict_res_perCate[line.cate][lv_iou][lv_score]['FP'] += 1

                            # FN should also be increased by 1, since this gt has no pred (diff cate)
                            dict_res[lv_iou][lv_score]['FN'] += 1
                            dict_res_perCate[line.cate][lv_iou][lv_score]['FN'] += 1
                    else:
                        dict_res[lv_iou][lv_score]['FN'] += 1
                        dict_res_perCate[line.cate][lv_iou][lv_score]['FN'] += 1
            else:
                for lv_score in LV_SCORE:
                    dict_res[lv_iou][lv_score]['FN'] += 1
                    dict_res_perCate[line.cate][lv_iou][lv_score]['FN'] += 1
    # -----------------------------------------------------------------------------

    # ------ loop through predicted bboxes to find FP (extra bboxes) ------
    for line in lines_pred_in:
        # modified to support no-semantic line segments from LSD:
        if (line.cate not in LABEL_MAP and line.cate != 'lineseg')\
                or (line.cate in LABEL2SKIP):
            continue

        # compare bbox with all ground truth bboxes
        # to find the one with largest IoU:
        max_iou = 0
        if_cate_right = False
        for line_gt in lines_gt_in:
            if line_gt.cate in LABEL2SKIP:
                continue
            iou = metric_func[metric](line_gt, line)
            # iou = _cal_acl_line(line_gt, line) if metric == 'acl' else _cal_iou_line(line_gt, line)
            if iou > max_iou:
                max_iou = iou
            if line.cate == line_gt.cate:
                if_cate_right = True
            else:
                if_cate_right = False

        # arrange output according to levels of IoU and levels of score to check:
        score_at_max_iou = line.score
        for lv_iou in LV_IOU:
            # if IoU smaller than current IoU threshold:
            if max_iou < lv_iou:
                for lv_score in LV_SCORE:
                    if score_at_max_iou > lv_score:
                        dict_res[lv_iou][lv_score]['FP'] += 1
                        if line.cate != 'lineseg':
                            dict_res_perCate[line.cate][lv_iou][lv_score]['FP'] += 1
    # ---------------------------------------------------------------------

    return dict_res, dict_res_perCate


def cal_metric_res_AngLen(
        lines_gt_in,
        lines_pred_in,
        metric='acl'
):
    # initiate output dict
    dict_res = OrderedDict((key_iou, 0) for key_iou in LV_IOU)
    for key_iou in dict_res.keys():
        dict_res[key_iou] = OrderedDict((key_score, 0) for key_score in LV_SCORE)
        for key_score in dict_res[key_iou].keys():
            dict_res[key_iou][key_score] = {
                'TP': 0,
                'TN': 0,
                'FP': 0,
                'FN': 0,
            }

    # initiate output dict with Cate level:
    dict_res_perCate = OrderedDict((key, 0) for key in LABEL_MAP if key not in LABEL2SKIP)
    for key_cate in dict_res_perCate.keys():
        dict_res_perCate[key_cate] = OrderedDict((key_iou, 0) for key_iou in LV_IOU)
        for key_iou in dict_res_perCate[key_cate].keys():
            dict_res_perCate[key_cate][key_iou] = OrderedDict((key_score, 0) for key_score in LV_SCORE)
            for key_score in dict_res_perCate[key_cate][key_iou].keys():
                dict_res_perCate[key_cate][key_iou][key_score] = {
                    'TP': 0,
                    'TN': 0,
                    'FP': 0,
                    'FN': 0,
                }

    # ------ loop through ground truth bboxes to find FN (missed bboxes) ------
    for line in lines_gt_in:
        # Each gt should be counted as TP or FN, this is to record if it has been recorded
        if_recorded = 0

        # Reivse some typo in prediction cates
        line.cate = 'ground_building' if line.cate == 'ground building' else line.cate

        if line.cate in LABEL2SKIP or line.cate not in LABEL_MAP:
            print("Found invalid line type: {}, skip.".format(line.cate))
            continue
        # compare bbox with all ground truth bboxes
        # to find the one with largest IoU:
        max_iou = 0
        if_cate_right = False
        score_at_max_iou = 0.
        for line_pred in lines_pred_in:
            # modified to support no-semantic line segments from LSD:
            if (line_pred.cate not in LABEL_MAP and line_pred.cate != 'lineseg')\
                    or (line_pred.cate in LABEL2SKIP):
                continue

            if line.cate in LABEL_MAP_OBJ_KAIST and line_pred.cate in LABEL_MAP_OBJ_KAIST:
                iou = _cal_iou_line(line, line_pred)
            else:
                iou = metric_func[metric](line, line_pred)
                # print(iou)
                # iou = _cal_acl_line(line, line_pred) if metric == 'acl' else _cal_iou_line(line, line_pred)
                # iou_test = _cal_acl_line_v1(line, line_pred)
                # print(iou_test)
            if iou > max_iou:
                max_iou = iou
                score_at_max_iou = line_pred.score
                if line.cate == line_pred.cate:
                    if_cate_right = True
                else:
                    if_cate_right = False if line_pred.cate != 'lineseg' else True

        # arrange output according to levels of IoU and levels of score to check:
        for lv_iou in LV_IOU:
            # if IoU larger than current IoU threshold:
            if max_iou > lv_iou:
                for lv_score in LV_SCORE:
                    # if score larger than current score threshold:
                    if score_at_max_iou > lv_score:
                        if if_cate_right:
                            dict_res[lv_iou][lv_score]['TP'] += 1
                            dict_res_perCate[line.cate][lv_iou][lv_score]['TP'] += 1
                        else:
                            # FP increase by 1, since this pred has no gt (diff cate)
                            dict_res[lv_iou][lv_score]['FP'] += 1
                            dict_res_perCate[line.cate][lv_iou][lv_score]['FP'] += 1

                            # FN should also be increased by 1, since this gt has no pred (diff cate)
                            dict_res[lv_iou][lv_score]['FN'] += 1
                            dict_res_perCate[line.cate][lv_iou][lv_score]['FN'] += 1
                    else:
                        dict_res[lv_iou][lv_score]['FN'] += 1
                        dict_res_perCate[line.cate][lv_iou][lv_score]['FN'] += 1
            else:
                for lv_score in LV_SCORE:
                    dict_res[lv_iou][lv_score]['FN'] += 1
                    dict_res_perCate[line.cate][lv_iou][lv_score]['FN'] += 1
    # -----------------------------------------------------------------------------

    # ------ loop through predicted bboxes to find FP (extra bboxes) ------
    for line in lines_pred_in:
        # modified to support no-semantic line segments from LSD:
        if (line.cate not in LABEL_MAP and line.cate != 'lineseg')\
                or (line.cate in LABEL2SKIP):
            continue

        # compare bbox with all ground truth bboxes
        # to find the one with largest IoU:
        max_iou = 0
        if_cate_right = False
        for line_gt in lines_gt_in:
            if line_gt.cate in LABEL2SKIP:
                continue
            iou = metric_func[metric](line_gt, line)
            # iou = _cal_acl_line(line_gt, line) if metric == 'acl' else _cal_iou_line(line_gt, line)
            if iou > max_iou:
                max_iou = iou
            if line.cate == line_gt.cate:
                if_cate_right = True
            else:
                if_cate_right = False

        # arrange output according to levels of IoU and levels of score to check:
        score_at_max_iou = line.score
        for lv_iou in LV_IOU:
            # if IoU smaller than current IoU threshold:
            if max_iou < lv_iou:
                for lv_score in LV_SCORE:
                    if score_at_max_iou > lv_score:
                        dict_res[lv_iou][lv_score]['FP'] += 1
                        if line.cate != 'lineseg':
                            dict_res_perCate[line.cate][lv_iou][lv_score]['FP'] += 1
    # ---------------------------------------------------------------------

    return dict_res, dict_res_perCate


def cal_mAP(dict_res_perCate_in):
    mAP = 0.
    mAP3 = 0.
    cnt_nonzero = 0
    for cate, sub_dict in dict_res_perCate_in.items():
        print("cate: ", cate)
        cate_AP = 0.
        cate_AP3 = 0.
        if_has_gt = False  # to check if there exist gt of this category
        for lv_iou in LV_IOU:
            TP = dict_res_perCate_in[cate][lv_iou][mAP_score]['TP']
            FP = dict_res_perCate_in[cate][lv_iou][mAP_score]['FP']
            FN = dict_res_perCate_in[cate][lv_iou][mAP_score]['FN']
            AP = TP / (TP + FP) if TP + FP > 0 else 0
            cate_AP += AP
            if cate in mAP_SUB:
                cate_AP3 += AP
            if_has_gt = True if TP + FN > 0 else False

            print("lv IoU: {} | TP: {} | FP: {} | FN : {} | AP: {} |".format(
                lv_iou, TP, FP, FN, AP
            ))

        cate_mAP = cate_AP/len(LV_IOU)
        cate_mAP3 = cate_AP3/len(LV_IOU)
        print("Category {} | AP {}".format(cate, cate_mAP))
        mAP += cate_mAP
        mAP3 += cate_mAP3
        if if_has_gt:
            cnt_nonzero += 1
            # print("... category {} has gt...".format(cate))

    print("# of None zero AP categories: ", cnt_nonzero)
    mAP = mAP / cnt_nonzero
    mAP3 = mAP3 / len(mAP_SUB)
    print("mAP @ score {} : ".format(mAP_score), mAP)
    print("mAP{} @ score {} : ".format(len(mAP_SUB), mAP_score), mAP3)
    return mAP, mAP3


def cal_mAP_mAR_F1(dict_res_perCate_in):
    mAP = 0.
    mAP3 = 0.
    mAR = 0.
    mAR3 = 0.
    mF1 = 0.
    mF13 = 0.
    cnt_nonzero = 0
    for cate, sub_dict in dict_res_perCate_in.items():
        print("cate: ", cate)
        cate_AP = 0.
        cate_AP3 = 0.
        cate_AR = 0.
        cate_AR3 = 0.
        cate_F1 = 0.
        cate_F13 = 0.
        if_has_gt = False  # to check if there exist gt of this category
        for lv_iou in LV_IOU:
            TP = dict_res_perCate_in[cate][lv_iou][mAP_score]['TP']
            FP = dict_res_perCate_in[cate][lv_iou][mAP_score]['FP']
            FN = dict_res_perCate_in[cate][lv_iou][mAP_score]['FN']
            AP = TP / (TP + FP) if TP + FP > 0 else 0
            AR = TP / (TP + FN) if TP + FN > 0 else 0
            F1 = AP * AR / (AP + AR) if AP > 0 and AR > 0 else 0
            cate_AP += AP
            cate_AR += AR
            cate_F1 += F1
            if cate in mAP_SUB:
                cate_AP3 += AP
                cate_AR3 += AR
                cate_F13 += F1
            if_has_gt = True if TP + FN > 0 else False

            print("lv IoU: {} | TP: {} | FP: {} | FN : {} | AP: {} |".format(
                lv_iou, TP, FP, FN, AP
            ))

        cate_mAP = cate_AP / len(LV_IOU)
        cate_mAP3 = cate_AP3 / len(LV_IOU)
        cate_mAR = cate_AR / len(LV_IOU)
        cate_mAR3 = cate_AR3 / len(LV_IOU)
        cate_F1 = cate_F1 / len(LV_IOU)
        cate_F13 = cate_F13 / len(LV_IOU)
        print("Category {} | AP {} | AR {} | F1 {}".format(cate, cate_mAP, cate_mAR, cate_F1))
        mAP += cate_mAP
        mAP3 += cate_mAP3
        mAR += cate_mAR
        mAR3 += cate_mAR3
        mF1 += cate_F1
        mF13 += cate_F13
        if if_has_gt:
            cnt_nonzero += 1
            # print("... category {} has gt...".format(cate))

    print("# of None zero AP categories: ", cnt_nonzero)
    mAP = mAP / cnt_nonzero
    mAP3 = mAP3 / len(mAP_SUB)
    mAR = mAR / cnt_nonzero
    mAR3 = mAR3 / len(mAP_SUB)
    mF1 = mF1 / cnt_nonzero
    mF13 = mF13 / len(mAP_SUB)
    print("mAP @ score {} : ".format(mAP_score), mAP)
    print("mAP{} @ score {} : ".format(len(mAP_SUB), mAP_score), mAP3)
    print("mAR @ score {} : ".format(mAP_score), mAR)
    print("mAR{} @ score {} : ".format(len(mAP_SUB), mAP_score), mAR3)
    print("F1 @ score {} : ".format(mAP_score), mF1)
    print("F1{} @ score {} : ".format(len(mAP_SUB), mAP_score), mF13)
    return mAP, mAP3


def performLineNMS(lineSegs_in):
    lst_line = [None] * len(lineSegs_in)
    for l, lineseg in enumerate(lineSegs_in):
        line = [lineseg.x_left, lineseg.y_left, lineseg.x_right, lineseg.y_right,
                lineseg.cate, lineseg.score, lineseg.direction]
        lst_line[l] = line
    dict_line_out = nms_func[METRIC](lst_line, thres_in=METRIC_THRESH)
    # dict_line_out = do_acl_line(lst_line, thres_in=METRIC_THRESH) if METRIC == 'acl' else \
    #    do_nms_line(lst_line, thres_in=METRIC_THRESH)

    lineSegs_out = []
    for cate, lines in dict_line_out.items():
        if lines is None:
            continue
        for line in lines:
            lineSeg_res = LineSeg(
                x_left_in=int(line[0]),
                y_left_in=int(line[1]),
                x_right_in=int(line[2]),
                y_right_in=int(line[3]),
                cate_in=cate,
                score_in=float(line[4]),
                direct_in=float(line[5])
            )
            lineSegs_out.append(lineSeg_res)

    return lineSegs_out


def eval():
    # initiate total result dict
    dict_total = OrderedDict((key_iou, 0) for key_iou in LV_IOU)
    for key_iou in dict_total.keys():
        dict_total[key_iou] = OrderedDict((key_score, 0) for key_score in LV_SCORE)
        for key_score in dict_total[key_iou].keys():
            dict_total[key_iou][key_score] = {
                'TP': 0,
                'TN': 0,
                'FP': 0,
                'FN': 0,
            }
    # initiate total result dict with Cate level:
    dict_total_perCate = OrderedDict((key, 0) for key in LABEL_MAP if key not in LABEL2SKIP)
    for key_cate in dict_total_perCate.keys():
        dict_total_perCate[key_cate] = OrderedDict((key_iou, 0) for key_iou in LV_IOU)
        for key_iou in dict_total_perCate[key_cate].keys():
            dict_total_perCate[key_cate][key_iou] = OrderedDict((key_score, 0) for key_score in LV_SCORE)
            for key_score in dict_total_perCate[key_cate][key_iou].keys():
                dict_total_perCate[key_cate][key_iou][key_score] = {
                    'TP': 0,
                    'TN': 0,
                    'FP': 0,
                    'FN': 0,
                }

    cnt_file = 0
    cnt_gt = 0
    for file in lst_gt:
        if 'TZKJ' in SRC_PRED:
            lines_pred = get_lines_from_xml(SRC_PRED + file)
        else:
            if file not in lst_pred:
                tmpfile = file.replace('val_semantic_line', '')
                if tmpfile not in lst_pred:
                    print("{} not found in prediction! Empty prediction...".format(file))
                    lines_pred = list()
                else:
                    lines_pred = get_lines_from_xml(SRC_PRED + tmpfile)
            else:
                lines_pred = get_lines_from_xml(SRC_PRED + file)

        lines_pred = performLineNMS(lines_pred)  # Line ACL
        lines_gt = get_lines_from_xml(SRC_GT + file, if_gt=True)
        cnt_gt += len(lines_gt)

        # metric_res, metric_res_perCate = cal_metric_res(lines_gt, lines_pred, metric=METRIC)
        metric_res, metric_res_perCate = cal_metric_res(lines_gt, lines_pred, metric=METRIC_EVAL)

        # update total results:
        for key_iou in dict_total.keys():
            for key_score in dict_total[key_iou].keys():
                dict_total[key_iou][key_score]['TP'] += metric_res[key_iou][key_score]['TP']
                dict_total[key_iou][key_score]['TN'] += metric_res[key_iou][key_score]['TN']
                dict_total[key_iou][key_score]['FP'] += metric_res[key_iou][key_score]['FP']
                dict_total[key_iou][key_score]['FN'] += metric_res[key_iou][key_score]['FN']

        for key_cate in LABEL_MAP:
            if key_cate in LABEL2SKIP:
              continue
            for key_iou in dict_total.keys():
                for key_score in dict_total[key_iou].keys():
                    dict_total_perCate[key_cate][key_iou][key_score]['TP'] += \
                        metric_res_perCate[key_cate][key_iou][key_score]['TP']
                    dict_total_perCate[key_cate][key_iou][key_score]['TN'] += \
                        metric_res_perCate[key_cate][key_iou][key_score]['TN']
                    dict_total_perCate[key_cate][key_iou][key_score]['FP'] += \
                        metric_res_perCate[key_cate][key_iou][key_score]['FP']
                    dict_total_perCate[key_cate][key_iou][key_score]['FN'] += \
                        metric_res_perCate[key_cate][key_iou][key_score]['FN']

        cnt_file += 1
        if cnt_file % 100 == 0:
            print("Checked {} images out of total {}...".format(cnt_file, len(lst_pred)))

    # Calculate mAP:
    print("Calculating mAP...")
    # mAP, mAP3 = cal_mAP(dict_res_perCate_in=dict_total_perCate)
    mAP, mAP3 = cal_mAP_mAR_F1(dict_res_perCate_in=dict_total_perCate)

    print("DONE, printing summary...")

    def cal_recall_precision(tp, tn, fp, fn):
        try:
            rec = float(tp) / float(tp + fn)
        except ZeroDivisionError:
            rec = float(tp) / (float(tp + fn) + 1e-6)
        try:
            prec = float(tp) / float(tp + fp)
        except:
            prec = float(tp) / (float(tp + fp) + 1e-6)
        return rec, prec

    print("For ROC curve:")
    txt4roc = ""
    for key_iou in dict_total.keys():
        TP = dict_total[key_iou][ROC_score]['TP']
        TN = dict_total[key_iou][ROC_score]['TN']
        FP = dict_total[key_iou][ROC_score]['FP']
        FN = dict_total[key_iou][ROC_score]['FN']
        recall, precision = cal_recall_precision(TP, TN, FP, FN)
        txt4roc += "{}|{}|".format(recall, precision)
    print(txt4roc)

    print("In total {} test images:".format(len(lst_gt)))
    for key_iou in dict_total.keys():
        print("IoU: {}".format(key_iou))
        for key_score in dict_total[key_iou].keys():
            print("\tConf: {}".format(key_score))

            TP = dict_total[key_iou][key_score]['TP']
            TN = dict_total[key_iou][key_score]['TN']
            FP = dict_total[key_iou][key_score]['FP']
            FN = dict_total[key_iou][key_score]['FN']
            num_obj = TP + TN + FN
            print("\tTotal gt objects: {}".format(num_obj))
            print("\tTP: {}\t TN: {}\t FP: {}\t FN:{}".format(TP, TN, FP, FN))
            recall, precision = cal_recall_precision(TP, TN, FP, FN)
            print("\tRecall: {:.4f}\t Precision: {:.4f}".format(recall, precision))

    with open(RECORD_TXT, 'w+') as res:
        res.write("mAP@score{}: {} | mAP{}@score{}: {}\n".format(mAP_score, mAP,
                                                               len(mAP_SUB), mAP_score, mAP3))
        res.write("IoU\tConfidence\tTP\tTN\tFP\tFN\tRecall\tPrecision\n")
        for key_iou in dict_total.keys():
            res.write("{}".format(key_iou))
            pre = '\t'
            for key_score in dict_total[key_iou].keys():
                res.write("{}{}".format(pre, key_score))
                pre = '\t'

                TP = dict_total[key_iou][key_score]['TP']
                TN = dict_total[key_iou][key_score]['TN']
                FP = dict_total[key_iou][key_score]['FP']
                FN = dict_total[key_iou][key_score]['FN']
                res.write("\t{}\t{}\t{}\t{}".format(TP, TN, FP, FN))
                recall, precision = cal_recall_precision(TP, TN, FP, FN)
                res.write("\t{:.4f}\t{:.4f}\n".format(recall, precision))

    with open(RECORD_perCate_TXT, 'w+') as res:
        res.write("Category\tIoU\tConfidence\tTP\tTN\tFP\tFN\tRecall\tPrecision\n")
        for key_cate in dict_total_perCate.keys():
            res.write("{}".format(key_cate))
            for key_iou in dict_total_perCate[key_cate].keys():
                pre = '\t\t'
                res.write("{}{}".format(pre, key_iou))
                pre = '\t'
                for key_score in dict_total_perCate[key_cate][key_iou].keys():
                    res.write("{}{}".format(pre, key_score))
                    pre = '\t\t\t'

                    TP = dict_total_perCate[key_cate][key_iou][key_score]['TP']
                    TN = dict_total_perCate[key_cate][key_iou][key_score]['TN']
                    FP = dict_total_perCate[key_cate][key_iou][key_score]['FP']
                    FN = dict_total_perCate[key_cate][key_iou][key_score]['FN']
                    res.write("\t{}\t{}\t{}\t{}".format(TP, TN, FP, FN))
                    recall, precision = cal_recall_precision(TP, TN, FP, FN)
                    res.write("\t{:.4f}\t{:.4f}\n".format(recall, precision))


if __name__ == '__main__':
    eval()
