import xml.etree.ElementTree as ET
from xml.dom import minidom
from collections import OrderedDict
import cv2
import json
import os
import numpy as np
import sys
import yaml

class_name_xixiyq = [
    'building', 'ground_building', 'pillar', 'wall', 'ground_wall',
    'stair', 'grass', 'fence', 'ground_fence', 'ceiling'
]
class_name_kitti = [
    'building', 'ground_building', 'wall', 'ground_wall',
    'grass', 'fence', 'ground_fence', 'pole', 'curb', 'sign', 'tree', 'window', 'door'
]
class_name_kaist = [
    'building', 'ground_building', 'wall', 'ground_wall',
    'grass', 'fence', 'ground_fence', 'pole', 'curb', 'sign', 'tree', 'window', 'door', 'bridge',
    'building_sky'
]
class_name_obj = [
    'Pedestrian', 'Cyclist', 'Car', 'Van', 'Truck', 'Tram',
    'rider', 'person', 'car', 'traffic_sign', 'road_light', 'road_light_area', 'traffic_light',
    'bus', 'motorcycle', 'bicycle',
    'personup', 'bicycledown', 'marker', 'marker_5',
]

LABEL_MAP = {
    1: 'building',
    2: 'ground_building',
    3: 'wall',
    4: 'ground_wall',
    5: 'grass',
    6: 'fence',
    7: 'ground_fence',
    8: 'pole',
    9: 'curb',
    10: 'sign',
    11: 'tree',
    12: 'window',
    13: 'door',
    14: 'bridge',
}

LABEL_MAP_REV = {
    'building': 0,
    'ground_building': 1,
    'wall': 2,
    'ground_wall': 3,
    'grass': 4,
    'fence': 5,
    'ground_fence': 6,
    'pole': 7,
    'curb': 8,
    'sign': 9,
    'tree': 10,
    'window': 11,
    'door': 12,
    'bridge': 13,
    'lineseg': 14
}


def checkPth(pth):
    if not os.path.exists(pth):
        os.makedirs(pth)


def visualize_progress(current_cnt, total_cnt):
    sys.stdout.write('\r>> Sampled %d/%d' % (
        current_cnt, total_cnt))
    sys.stdout.flush()


# Fix some typos in labels
def fix_typos(str_in):
    str_out = str_in.replace('-', '_')  # replace '-' (if any) with '_'
    str_out = str_out.replace('glass', 'grass')  # replace 'glass' (if any) with 'grass'
    str_out = str_out.replace('building\\', 'building')  # replace 'glass' (if any) with 'grass'

    return str_out


# prettify the output format of xml
def prettify(elem, doctype = None):
    """Return a pretty-printed XML string for the Element.
        """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    if doctype is not None:
        reparsed.insertBefore(doctype, reparsed.documentElement)
    return reparsed.toprettyxml(indent="  ")


# To sort point coordinates in labeme resultant .json files.
# Input:
#   A list of two points. First element is [x, y] of first point, second element is [x, y] of the second point.
# Output:
#   A list with sorted point coordinates, [x1, y1, x2, y2],
#   where x1, y1 is the coordinate of the left point. If two points have same x values, then x1, y1 is the upper point.


def sortPoint(lst_point):
    assert isinstance(lst_point, list), print("Input points must be a list!")
    try:
        assert len(lst_point) == 2, print("Input points list must be in format lst[2][2]"
                                          "where lst[0][0] and lst[0][1] is point1 x, y,"
                                          "lst[1][0] and lst[1][1] is point2 x, y")
    except AssertionError:
        print(lst_point)

    if lst_point[0][0] < lst_point[1][0]:
        x1 = lst_point[0][0]
        y1 = lst_point[0][1]
        x2 = lst_point[1][0]
        y2 = lst_point[1][1]
    else:
        x1 = lst_point[1][0]
        y1 = lst_point[1][1]
        x2 = lst_point[0][0]
        y2 = lst_point[0][1]

    cord_res = [x1, y1, x2, y2]

    return cord_res


def _build_cate_dict(class_name):
    dict_category = list()
    for i in range(len(class_name)):
        dict_category.append({"supercategory": "none", "id": i+1, "name": class_name[i]})
    return dict_category


# Load labelme .json
def loadJsons2lines(json_root, class2skip=None, dataset='xixiyq', if_sort=True):
    lst_json = sorted([f for f in os.listdir(json_root) if '.json' in f])

    attrDict = OrderedDict()
    if dataset == 'xixiyq':
        attrDict['categories'] = _build_cate_dict(class_name_xixiyq)
    elif dataset == 'kaist':
        attrDict['categories'] = _build_cate_dict(class_name_kaist)
    else:
        attrDict['categories'] = _build_cate_dict(class_name_kitti)

    img_id = 0
    mid_res = OrderedDict()
    for json_file in lst_json:
        with open(json_root + json_file, 'r') as j_in:
            json_dict = OrderedDict(json.load(j_in))

        mid_format = dict()
        # img_dict['file_name'] = json_dict['imagePath']
        mid_format['file_name'] = json_file[:-5] + json_dict['imagePath'][-4:]
        mid_format['height'] = json_dict['imageHeight']
        mid_format['width'] = json_dict['imageWidth']
        mid_format['id'] = img_id
        mid_format['lines'] = list()

        for shape in json_dict['shapes']:

            shape['label'] = shape['label'].replace('-', '_')  # replace '-' (if any) with '_'
            shape['label'] = shape['label'].replace('glass', 'grass')  # replace 'glass' (if any) with 'grass'
            shape['label'] = shape['label'].replace('building\\', 'building')  # replace 'glass' (if any) with 'grass'

            if class2skip is not None:
                if shape['label'] in class2skip:
                    continue
            try:
                if shape['shape_type'] == 'polygon' or \
                        (len(shape['points']) > 2 and shape['shape_type'] == 'line'):
                    lst_pt_pairs = list()
                    for i in range(len(shape['points']) - 1):
                        pts = [shape['points'][i], shape['points'][i + 1]]
                        lst_pt_pairs.append(pts)
                    lst_pt_pairs.append([shape['points'][-1], shape['points'][0]])
                else:
                    lst_pt_pairs = [shape['points']]
            except KeyError:
                if len(shape['points']) > 2:
                    lst_pt_pairs = list()
                    for i in range(len(shape['points']) - 1):
                        pts = [shape['points'][i], shape['points'][i + 1]]
                        lst_pt_pairs.append(pts)
                    lst_pt_pairs.append([shape['points'][-1], shape['points'][0]])
                else:
                    lst_pt_pairs = [shape['points']]

            shape_id = 0
            for pts in lst_pt_pairs:
                anno_dict = dict()
                anno_dict['iscrowd'] = 0
                anno_dict['image_id'] = img_id
                if len(pts) == 4:
                    print("pause")
                if if_sort:
                    sorted_cord = sortPoint(pts)
                else:
                    sorted_cord = list()
                    for pt in pts:
                        sorted_cord.append(pt[0])
                        sorted_cord.append(pt[1])
                anno_dict.update({'cord': sorted_cord})
                anno_dict['area'] = 0
                category_id = 0
                for cate in attrDict['categories']:
                    if cate['name'] == shape['label']:
                        category_id = cate['id']
                        break
                if category_id == 0:
                    print("pause, category id 0")
                anno_dict['category_id'] = category_id
                anno_dict['ignore'] = 0
                anno_dict['id'] = shape_id
                try:
                    anno_dict['score'] = shape['score']
                except KeyError:
                    anno_dict['score'] = 1.0
                shape_id += 1

                mid_format['lines'].append(anno_dict)

        img_id += 1
        mid_res[json_file] = mid_format

    return mid_res


def parseXml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    bboxes = []
    size = root.find('size')
    if size is None:
        return 0, 0, bboxes
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        xmin = int(float(bndbox.find('xmin').text))
        ymin = int(float(bndbox.find('ymin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymax = int(float(bndbox.find('ymax').text))

        name = obj.find('name').text
        pose = obj.find('pose').text
        difficult = obj.find('difficult').text
        truncated = obj.find('truncated').text
        try:
            direction = obj.find('direction').text
        except AttributeError:
            direction = None
        try:
            score = obj.find('score').text
        except AttributeError:
            score = 1.0

        if direction is not None:
            bbox = [xmin, ymin, xmax, ymax, name, pose, difficult, truncated, direction, score]
        else:
            bbox = [xmin, ymin, xmax, ymax, name, pose, difficult, truncated, score]
        bboxes.append(bbox)

    return width, height, bboxes


# Input lines:
#   [xmin, ymin, xmax, ymax, name, score, direction]
def write_xml(dst_pth, dst_name, height, width, lines, folder_in=None, filename_in=None, path_in=None):
    root = ET.Element("annotation")
    folder = ET.SubElement(root, "folder")
    folder.text = folder_in
    filename = ET.SubElement(root, "filename")
    filename.text = filename_in

    path = ET.SubElement(root, "path")
    path.text = path_in

    h, w, c = height, width, 3

    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(w)
    ET.SubElement(size, "height").text = str(h)
    ET.SubElement(size, "depth").text = str(c)

    segmented = ET.SubElement(root, "segmented")
    segmented.text = "0"

    for line in lines:
        object_box = ET.SubElement(root, "object")
        ET.SubElement(object_box, "name").text = line[4]
        ET.SubElement(object_box, "score").text = str(line[5])

        ET.SubElement(object_box, "pose").text = "Unspecified"
        ET.SubElement(object_box, "truncated").text = "0"
        ET.SubElement(object_box, "difficult").text = "0"

        bndbox = ET.SubElement(object_box, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(line[0])
        ET.SubElement(bndbox, "ymin").text = str(line[1])
        ET.SubElement(bndbox, "xmax").text = str(line[2])
        ET.SubElement(bndbox, "ymax").text = str(line[3])
        ET.SubElement(object_box, "direction").text = line[-1]

    xmlstr = prettify(root)
    xml_dst = dst_pth + dst_name
    with open(xml_dst, 'w+') as f:
        f.write(xmlstr)


def correct_cords(xleft, yleft, xright, yright, img_w=1280, img_h=800):
    if xleft == xright:
        alpha = 1.0
        beta = 0.
    else:
        alpha = (yright - yleft) / (xright - xleft)
        beta = (xright * yleft - xleft * yright) / (xright - xleft)

    wlimit = img_w - 1
    hlimit = img_h - 1

    # Skip lines outside image.
    if (xleft < 0 or xright > wlimit) and xleft == xright:
        xright = np.clip(xright, 0, wlimit)
        xleft = np.clip(xleft, 0, wlimit)
    if (yleft < 0 or yright > hlimit) and yleft == yright:
        yright = np.clip(yright, 0, hlimit)
        yleft = np.clip(yleft, 0, hlimit)
        # return None

    #while xleft < 0 or xright < 0 or xleft > wlimit or xright > wlimit \
    #    or yleft < 0 or yright < 0 or yleft > hlimit or yright > hlimit:
    # Correct X first.
    if xleft < 0:
        xleft = 0
        yleft = beta
    if xright > wlimit:
        xright = wlimit
        yright = alpha * xright + beta
    # Correct Y next.
    if yleft < 0:
        yleft = 0
        xleft = -beta / alpha if alpha != 0 else xleft
    if yright < 0:
        yright = 0
        xright = -beta / alpha if alpha != 0 else xright
    if yleft > hlimit:
        yleft = hlimit
        xleft = (yleft - beta) / alpha if alpha != 0 else xleft
    if yright > hlimit:
        yright = hlimit
        xright = (yright - beta) / alpha if alpha != 0 else xright

    # Correct X first.
    if xleft > wlimit:
        xleft = wlimit
        yleft = alpha * xleft + beta
    if xright > wlimit:
        xright = wlimit
        yright = alpha * xright + beta
    return [xleft, yleft, xright, yright]


def write_yaml(dict_all_lines_in, yaml_dst=None,
               rectifier_model="mynt", rectifier=None,
               img_pth_distorted=None,
               func_distorter_mynt=None, func_distorter_kaist=None):

    dict_all_dets = dict_all_lines_in

    yaml_dict = dict()
    yaml_dict['images'] = list()
    cnt = 0
    cnt_det = 0
    for img_name, dets in dict_all_dets.items():
        cnt_det += len(dets['lines'])
        # lines2draw = []
        # lines2draw_rectified = []
        if 'json' in img_name:
            timestamp = img_name[:-5]
        elif '.xml' in img_name:
            timestamp = img_name[:-4]
        else:
            timestamp = img_name
        tmp_dict = dict()
        data_str = list()

        if len(dets['lines']) == 0:
            tmp_dict['timestamp'] = int(timestamp)
            tmp_dict['line'] = {
                'row': len(dets['lines']),
                'col': 6,
                'data': '',
            }
            yaml_dict['images'].append({'image': tmp_dict})
            cnt += 1
            continue

        pts = np.ones((2, 2*len(dets['lines'])))
        for d, det in enumerate(dets['lines']):
            x_left = det['cord'][0]
            y_left = det['cord'][1]
            x_right = det['cord'][2]
            y_right = det['cord'][3]

            res = correct_cords(x_left, y_left, x_right, y_right)
            if res is not None:
                [x_left, y_left, x_right, y_right] = res
            else:
                continue
            # tmp_lst = [x_left, y_left, x_right, y_right, LABEL_MAP[det[4]], float(det[5])]
            # data_str += tmp_lst
            pts[0][d * 2] = x_left
            pts[1][d * 2] = y_left
            pts[0][d * 2 + 1] = x_right
            pts[1][d * 2 + 1] = y_right

            # line2draw_rectified = OrderedDict()
            # line2draw_rectified['x_left'] = x_left
            # line2draw_rectified['y_left'] = y_left
            # line2draw_rectified['x_right'] = x_right
            # line2draw_rectified['y_right'] = y_right
            # line2draw_rectified['cate'] = LABEL_MAP[det['category_id']]
            # lines2draw_rectified.append(line2draw_rectified)

        # Converting point coordinates to distorted coordinates.
        if rectifier_model == "wyxl" and rectifier is not None:
            pts_distorted = func_distorter_mynt(rectifier, pts)
        elif rectifier_model == "kaist" and rectifier is not None:
            pts_distorted = func_distorter_kaist(rectifier, pts)
        else:
            pts_distorted = pts

        for d, det in enumerate(dets['lines']):
            x_left = pts_distorted[0][d*2]
            y_left = pts_distorted[1][d*2]
            x_right = pts_distorted[0][d*2+1]
            y_right = pts_distorted[1][d*2+1]
            tmp_lst = [x_left, y_left, x_right, y_right, det['category_id'], det['score']]
            if det['category_id'] > 15:
                print("Found id > 14!\n{}".format(tmp_lst))
            data_str += tmp_lst
            # line2draw = OrderedDict()
            # line2draw['x_left'] = x_left
            # line2draw['y_left'] = y_left
            # line2draw['x_right'] = x_right
            # line2draw['y_right'] = y_right
            # line2draw['cate'] = LABEL_MAP[det['category_id']]
            # lines2draw.append(line2draw)

        # Visualization on distorted images.
        # if img_pth_distorted is not None:
        #     try:
        #         img = cv2.imread(img_pth_distorted + img_name[:-5] + '.png')
        #         if img is not None:
        #             img_drawn = draw_lines(img_in=img, lines_in=lines2draw)
        #             img_rectified_drawn = draw_lines(img_in=img, lines_in=lines2draw_rectified)
        #             cv2.imshow('vis distorted det on img', img_drawn)
        #             cv2.imshow('vis original det on img', img_rectified_drawn)
        #             cv2.waitKey(0)
        #     except FileNotFoundError:
        #         pass

        data_arr = np.asarray(data_str).astype(np.float).tolist()
        tmp_dict['timestamp'] = int(timestamp)
        tmp_dict['line'] = {
            'row': len(dets['lines']),
            'col': 6,
            'data': data_arr,
        }
        yaml_dict['images'].append({'image': tmp_dict})

        sys.stdout.write('\r>> Converting image %d/%d' % (
            cnt + 1, len(dict_all_dets)))
        sys.stdout.flush()
        cnt += 1

    avg_mnt_det = float(cnt_det) / float(len(dict_all_dets))
    print("\nAverage amount of detected Sem-LS: {}".format(avg_mnt_det))

    with open(yaml_dst, 'w+') as y_out:
        yaml.safe_dump(yaml_dict, y_out, default_flow_style=False)


# return the acl value from given line
def acl_line(det, gts):
    x1 = det[0]
    y1 = det[1]
    x2 = det[2]
    y2 = det[3]
    scores = det[4]
    directions = det[5]

    x_center = (x2 + x1) / 2
    y_center = (y2 + y1) / 2
    length = np.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
    x_intersection = x2 - x1
    y_intersection = y2 - y1
    x_intersection = 1.0 if x_intersection == 0. else x_intersection
    alpha = np.abs(np.arctan(y_intersection / x_intersection) * 180 / 3.14159265359)

    max_acl = 0.
    max_gt = None
    for gt in gts:
        ix_c = (gt[0] + gt[2]) / 2
        iy_c = (gt[1] + gt[3]) / 2
        ilength = np.sqrt((gt[2] - gt[0]) * (gt[2] - gt[0]) + (gt[3] - gt[1]) * (gt[3] - gt[1]))
        ix_inter = gt[2] - gt[0]
        iy_inter = gt[3] - gt[1]
        ix_inter = 1.0 if ix_inter == 0 else ix_inter
        ialpha = np.abs(np.arctan(iy_inter / ix_inter) * 180 / 3.14159265359)

        sim_a = max(0, 1 - (np.abs(ialpha - alpha) * 0.011111111111))
        sim_c = max(0, 1 - (np.sqrt((x_center - ix_c) * (x_center - ix_c) + (y_center - iy_c) * (y_center - iy_c)))
                    / (ilength * 0.5))
        sim_l = max(0, 1 - np.abs(ilength - length) / ilength)
        acl = sim_a * sim_c * sim_l

        max_acl = acl if acl > max_acl else max_acl
        max_gt = gt if acl > max_acl else max_gt

    return max_acl, max_gt


def get_lines_from_xml(xml_file, if_gt=False):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    lst_lines = list()
    for obj in root.findall('object'):
        objline = obj.find('bndbox')

        name = obj.find('name').text
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
        if name == 'lineseg':
            x_left = xmin
            y_left = ymin
            x_right = xmax
            y_right = ymax
        else:
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

        line = OrderedDict()
        line['x_left'] = int(x_left)
        line['y_left'] = int(y_left)
        line['x_right'] = int(x_right)
        line['y_right'] = int(y_right)
        line['cate'] = name
        line['score'] = float(score)
        line['direction'] = float(direction)
        lst_lines.append(line)

    return lst_lines


def draw_lines(img_in, lines_in):
    colors = gen_color_list()
    img_res = img_in.copy()
    for line in lines_in:
        x_left = int(line['x_left'])
        y_left = int(line['y_left'])
        x_right = int(line['x_right'])
        y_right = int(line['y_right'])
        name = line['cate']
        name = name.replace('ground building', 'ground_building')
        cat = LABEL_MAP_REV[name]

        c = colors[cat][0][0].tolist()
        txt = '{}'.format(name)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
        cv2.line(img_res, (x_left, y_left), (x_right, y_right), c, thickness=2)
        if name != 'lineseg':
            cv2.rectangle(img_res,
                          (x_left, y_left - cat_size[1] - 2),
                          (x_left + cat_size[0], y_left - 2), c, -1)
            cv2.putText(img_res, txt, (x_left, y_left - 2),
                        font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

    return img_res


def gen_color_list():
    color_list = np.array(
            [
                1.000, 0.333, 1.000,
                0.850, 0.325, 0.098,
                0.929, 0.694, 0.125,
                0.494, 0.184, 0.556,
                0.466, 0.674, 0.188,
                0.301, 0.745, 0.933,
                0.635, 0.078, 0.184,
                0.300, 0.700, 0.800,
                0.600, 0.600, 0.600,
                1.000, 0.000, 0.000,
                1.000, 0.500, 0.000,
                0.749, 0.749, 0.000,
                0.000, 1.000, 0.000,
                0.000, 0.000, 1.000,
                0.333, 1.000, 0.100,
                0.333, 0.333, 0.000,
                0.333, 0.667, 0.000,
                0.333, 1.000, 0.000,
                0.667, 0.333, 0.000,
                0.667, 0.667, 0.000,
                0.667, 1.000, 0.000,
                1.000, 0.333, 0.000,
                1.000, 0.667, 0.000,
                1.000, 1.000, 0.000,
                0.000, 0.333, 0.500,
                0.000, 0.667, 0.500,
                0.000, 1.000, 0.500,
                0.333, 0.000, 0.500,
                0.333, 0.333, 0.500,
                0.333, 0.667, 0.500,
                0.333, 1.000, 0.500,
                0.667, 0.000, 0.500,
                0.667, 0.333, 0.500,
                0.667, 0.667, 0.500,
                0.667, 1.000, 0.500,
                1.000, 0.000, 0.500,
                1.000, 0.333, 0.500,
                1.000, 0.667, 0.500,
                1.000, 1.000, 0.500,
                0.000, 0.333, 1.000,
                0.000, 0.667, 1.000,
                0.000, 1.000, 1.000,
                0.333, 0.000, 1.000,
                0.333, 0.333, 1.000,
                0.333, 0.667, 1.000,
                0.333, 1.000, 1.000,
                0.667, 0.000, 1.000,
                0.667, 0.333, 1.000,
                0.667, 0.667, 1.000,
                0.667, 1.000, 1.000,
                1.000, 0.000, 1.000,
                1.000, 0.333, 1.000,
                1.000, 0.667, 1.000,
                0.167, 0.000, 0.000,
                0.333, 0.000, 0.000,
                0.500, 0.000, 0.000,
                0.667, 0.000, 0.000,
                0.833, 0.000, 0.000,
                1.000, 0.000, 0.000,
                0.000, 0.167, 0.000,
                0.000, 0.333, 0.000,
                0.000, 0.500, 0.000,
                0.000, 0.667, 0.000,
                0.000, 0.833, 0.000,
                0.000, 1.000, 0.000,
                0.000, 0.000, 0.167,
                0.000, 0.000, 0.333,
                0.000, 0.000, 0.500,
                0.000, 0.000, 0.667,
                0.000, 0.000, 0.833,
                0.000, 0.000, 1.000,
                0.000, 0.000, 0.000,
                0.143, 0.143, 0.143,
                0.286, 0.286, 0.286,
                0.429, 0.429, 0.429,
                0.571, 0.571, 0.571,
                0.714, 0.714, 0.714,
                0.857, 0.857, 0.857,
                0.000, 0.447, 0.741,
                0.50, 0.5, 0
            ]
        ).astype(np.float32)
    color_list = color_list.reshape((-1, 3)) * 255
    colors = [(color_list[_]).astype(np.uint8) \
              for _ in range(len(color_list))]
    colors = np.array(colors, dtype=np.uint8).reshape(len(colors), 1, 1, 3)

    return colors