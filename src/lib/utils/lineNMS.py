from collections import OrderedDict
import numpy as np

from external_nms.nms import nms_line

NAMES = [
    'building', 'ground_building', 'pillar', 'wall', 'ground_wall',
    'stair', 'grass', 'fence', 'ground_fence', 'ceiling',
    'pole', 'curb', 'window', 'tree', 'sign', 'door', 'bridge',
    'car', 'person', 'rider', 'bus', 'traffic_sign', 'traffic_light', 'road_light',
    'road_light_area', 'motorcycle', 'bicycle',
    'cyclist', 'van', 'truck', 'tram',
    'lineseg',
]


def _reformulate_line_dets(lst_line_in):
    dict_line_per_cate = OrderedDict()
    for cate in NAMES:
        dict_line_per_cate[cate] = None

    for l in range(len(lst_line_in)):
        array_line = np.zeros(6)
        array_line[0] = lst_line_in[l][0]
        array_line[1] = lst_line_in[l][1]
        array_line[2] = lst_line_in[l][2]
        array_line[3] = lst_line_in[l][3]
        array_line[4] = lst_line_in[l][5]
        array_line[5] = 0 if lst_line_in[l][6] == 'lt2rb' else 1
        if dict_line_per_cate[lst_line_in[l][4]] is None:
            dict_line_per_cate[lst_line_in[l][4]] = [array_line.astype(np.float32)]
        else:
            dict_line_per_cate[lst_line_in[l][4]].append(array_line.astype(np.float32))

    return dict_line_per_cate


def do_nms_line(dict_line_per_cate_in, thres_in=0.9):
    dict_line_per_cate_out = dict_line_per_cate_in.copy()

    dict_line_per_cate_out = _reformulate_line_dets(dict_line_per_cate_out)

    for cate in NAMES:
        if dict_line_per_cate_out[cate] is None or len(dict_line_per_cate_out[cate]) == 1:
            continue
        keep_idx = nms_line(np.asarray(dict_line_per_cate_out[cate]), thres_in)
        # print(cate, keep_idx)

        if len(keep_idx) < len(dict_line_per_cate_out[cate]):
            dict_line_per_cate_out[cate] = [dict_line_per_cate_out[cate][x] for x in keep_idx]

    return dict_line_per_cate_out


def _fast_reject(line1_xleft, line1_yleft, line1_xright, line1_yright,
                 line2_xleft, line2_yleft, line2_xright, line2_yright):
    if line1_xleft > line2_xright or line1_xright < line2_xleft:
        return True
    line1_ymin = min(line1_yleft, line1_yright)
    line1_ymax = max(line1_yleft, line1_yright)
    line2_ymin = min(line2_yleft, line2_yright)
    line2_ymax = max(line2_yleft, line2_yright)
    if line1_ymin > line2_ymax or line1_ymax < line2_ymin:
        return True
    return False


def _acl_line(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    directions = dets[:, 5]

    x_centers = (x2 + x1) / 2
    y_centers = (y2 + y1) / 2
    lengths = np.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
    x_intersections = x2 - x1
    y_intersections = y2 - y1
    x_intersections[x_intersections == 0] = 1.0
    alphas = np.abs(np.arctan(y_intersections / x_intersections) * 180 / 3.14159265359)

    order = scores.argsort()[::-1]
    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int)

    keep = []
    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        ix_c = x_centers[i]
        iy_c = y_centers[i]
        ilength = lengths[i]
        ialpha = alphas[i]
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue

            # if _fast_reject(x1[i], y1[i], x2[i], y2[i],
            #                x1[j], y1[j], x2[j], y2[j]):
            #    acl = 0.
            # else:
            if directions[i] != directions[j]:
                acl = 0.
            else:
                jx_c = x_centers[j]
                jy_c = y_centers[j]
                jlength = lengths[j]
                jalpha = alphas[j]

                sim_a = max(0, 1 - (np.abs(ialpha - jalpha) * 0.011111111111))
                sim_c = max(0, 1 - (np.sqrt((jx_c - ix_c) * (jx_c - ix_c) +
                                            (jy_c - iy_c) * (jy_c - iy_c))) / (ilength * 0.5))
                sim_l = max(0, 1 - np.abs(ilength - jlength) / ilength)
                acl = sim_a * sim_c * sim_l

            if acl >= thresh and directions[i] == directions[j]:
                suppressed[j] = 1

    return keep


def _gaussian_radius(height, width, min_overlap=0.7):
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2
    return r1


def _acl_line_v1(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    directions = dets[:, 5]

    x_centers = (x2 + x1) / 2
    y_centers = (y2 + y1) / 2
    lengths = np.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
    x_intersections = x2 - x1
    y_intersections = y2 - y1
    x_intersections[x_intersections == 0] = 1.0

    order = scores.argsort()[::-1]
    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int)

    keep = []
    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        ix_c = x_centers[i]
        iy_c = y_centers[i]
        iw = x_intersections[i]
        ih = y_intersections[i]
        ilength = lengths[i]
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            jx_c = x_centers[j]
            jy_c = y_centers[j]
            jw = x_intersections[j]
            jh = y_intersections[j]
            jlength = lengths[j]

            radius = _gaussian_radius(ih, iw)
            sigma = (2 * radius - 1) / 6
            d_x = ix_c - jx_c
            d_y = iy_c - jy_c

            sim_a = np.abs((iw * jw + ih * jh) / (ilength * jlength))
            sim_c = np.exp(-(d_x * d_x + d_y * d_y) / (2 * sigma * sigma))
            sim_l = max(0, 1 - np.abs(ilength - jlength) / ilength)
            acl = sim_a * sim_c * sim_l

            if acl >= thresh and directions[i] == directions[j]:
                suppressed[j] = 1

    return keep


def do_acl_line(dict_line_per_cate_in, thres_in=0.7):
    dict_line_per_cate_out = dict_line_per_cate_in.copy()

    dict_line_per_cate_out = _reformulate_line_dets(dict_line_per_cate_out)

    for cate in NAMES:
        if dict_line_per_cate_out[cate] is None or len(dict_line_per_cate_out[cate]) == 1:
            continue
        keep_idx = _acl_line(np.asarray(dict_line_per_cate_out[cate]), thres_in)
        # print(cate, keep_idx)

        if len(keep_idx) < len(dict_line_per_cate_out[cate]):
            dict_line_per_cate_out[cate] = [dict_line_per_cate_out[cate][x] for x in keep_idx]
        else:
            pass

    return dict_line_per_cate_out


def do_acl_line_v1(dict_line_per_cate_in, thres_in=0.7):
    dict_line_per_cate_out = dict_line_per_cate_in.copy()

    dict_line_per_cate_out = _reformulate_line_dets(dict_line_per_cate_out)

    for cate in NAMES:
        if dict_line_per_cate_out[cate] is None or len(dict_line_per_cate_out[cate]) == 1:
            continue
        keep_idx = _acl_line_v1(np.asarray(dict_line_per_cate_out[cate]), thres_in)
        # print(cate, keep_idx)

        if len(keep_idx) < len(dict_line_per_cate_out[cate]):
            dict_line_per_cate_out[cate] = [dict_line_per_cate_out[cate][x] for x in keep_idx]
        else:
            pass

    return dict_line_per_cate_out
