# To load labelme .json files and convert to COCO .json format
# 'bbox' won't be filled, but fille to 'cord' attribute instead.
# Input:
#   json_root: path to .json files from labelm software
# Output:
#   dst_root: output path to the COCO-like .json file

import json
import os
from collections import OrderedDict
from utils import sortPoint

phase = 'train'
# json_roots = ['/Users/yisun/Desktop/Robotics_SemanticLines/data/bags/20191003_bags_1705_1714_combined/images/'
#              '{}_semantic_line/'.format(phase)]
# json_roots = ['/workspace/tangyang.sy/Robotics_SemanticLines/data/KAIST_labeled/KAIST_0903_train_val/images/'
#               '{}_semantic_line/'.format(phase)]
json_roots = ['/Users/yisun/Desktop/Robotics_SemanticLines/data/KAIST_URBAN/20200115_zhongbao/images/'
             '{}_semantic_line_json/'.format(phase)]
# dst_root = '/workspace/tangyang.sy/Robotics_SemanticLines/data/KAIST_labeled/KAIST_0903_train_val/images/annotations/'
dst_root = '/Users/yisun/Desktop/Robotics_SemanticLines/data/KAIST_URBAN/20200115_zhongbao/annotations/'
dst_name = '{}_semantic_line.json'.format(phase)
dataset = 'semantic_line_kaist'

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
    'grass', 'fence', 'ground_fence', 'pole', 'curb', 'sign', 'tree', 'window', 'door',
    'bridge'
]
class_name_obj_line = [
    'building', 'ground_building', 'wall', 'ground_wall',
    'grass', 'fence', 'ground_fence', 'pole', 'curb', 'sign', 'tree', 'window', 'door',
    'bridge',
    'car', 'person', 'rider', 'bus', 'traffic_sign', 'traffic_light', 'road_light',
    'road_light_area', 'motorcycle', 'bicycle'
]
class_name_obj_line_kitti = [
    'building', 'ground_building', 'wall', 'ground_wall',
    'grass', 'fence', 'ground_fence', 'pole', 'curb', 'sign', 'tree', 'window', 'door', 'bridge',
    'person', 'Cyclist', 'car', 'Van', 'Truck', 'Tram',
]
class_name_obj_kaist = [
    'car', 'person', 'rider', 'bus', 'traffic_sign', 'traffic_light', 'road_light',
    'road_light_area', 'motorcycle', 'bicycle'
]
class_name_obj_kitti = [
    'person', 'Cyclist', 'car', 'Van', 'Truck', 'Tram',
]
class2skip = [
    # 'sign', 'window', 'tree'
]

MapDataset2Anno = {
    'semantic_line_xixiyq': class_name_xixiyq,
    'semantic_line_kitti': class_name_kitti,
    'semantic_line_kaist': class_name_kaist,
    'obj_line': class_name_obj_line,
    'obj_line_kitti': class_name_obj_line_kitti,
    'obj_kaist': class_name_obj_kaist,
    'obj_kitti': class_name_obj_kitti,
}


def _build_cate_dict(class_name):
    dict_category = list()
    for i in range(len(class_name)):
        dict_category.append({"supercategory": "none", "id": i+1, "name": class_name[i].lower()})
    return dict_category


def labelme2cocolike(json_roots, dst_root, dst_name="labelme2coco.json", dataset='semantic_line_kaist'):

    print("\n*******************CLASS TO SKIP***************************")
    print(class2skip)
    print("***********************************************************")

    if not os.path.exists(dst_root):
        os.makedirs(dst_root)

    attrDict = OrderedDict()
    attrDict['categories'] = _build_cate_dict(MapDataset2Anno[dataset])

    lst_all_imgs = list()
    lst_all_annos = list()
    cnt_bbox = 0
    for json_root in json_roots:

        lst_json = sorted([f for f in os.listdir(json_root) if '.json' in f])

        img_id = 0
        shape_id = 1
        for json_file in lst_json:
            with open(json_root + json_file, 'r') as j_in:
                json_dict = OrderedDict(json.load(j_in))

            img_dict = dict()
            # img_dict['file_name'] = json_dict['imagePath']
            img_dict['file_name'] = json_file[:-5] + json_dict['imagePath'][-4:]
            img_dict['height'] = json_dict['imageHeight']
            img_dict['width'] = json_dict['imageWidth']
            img_dict['id'] = img_id

            lst_all_imgs.append(img_dict)

            for shape in json_dict['shapes']:

                shape['label'] = shape['label'].replace('-', '_')  # replace '-' (if any) with '_'
                shape['label'] = shape['label'].replace('glass', 'grass')  # replace 'glass' (if any) with 'grass'
                shape['label'] = shape['label'].replace('building\\', 'building')
                shape['label'] = shape['label'].replace('ground building',
                                                        'ground_building')

                isbbox = False
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
                        if shape['shape_type'] == 'rectangle':
                            # print(shape)
                            isbbox = True
                            cnt_bbox += 1
                except KeyError:
                    if len(shape['points']) > 2:
                        lst_pt_pairs = list()
                        for i in range(len(shape['points']) - 1):
                            pts = [shape['points'][i], shape['points'][i + 1]]
                            lst_pt_pairs.append(pts)
                        lst_pt_pairs.append([shape['points'][-1], shape['points'][0]])
                    else:
                        lst_pt_pairs = [shape['points']]

                for pts in lst_pt_pairs:
                    anno_dict = dict()
                    anno_dict['iscrowd'] = 0
                    anno_dict['image_id'] = img_id
                    if len(pts) == 4:
                        print("pause")
                    sorted_cord = sortPoint(pts)
                    anno_dict.update({'cord': sorted_cord})
                    anno_dict['area'] = 0 if not isbbox else 1
                    category_id = 0
                    for cate in attrDict['categories']:
                        if cate['name'] == shape['label'].lower():
                            category_id = cate['id']
                            break
                    anno_dict['category_id'] = category_id
                    anno_dict['ignore'] = 0
                    anno_dict['id'] = shape_id
                    lst_all_annos.append(anno_dict)
                    shape_id += 1

            img_id += 1

    attrDict["images"] = lst_all_imgs
    attrDict["annotations"] = lst_all_annos
    attrDict["type"] = "instances"

    print("Converted:\nImages {} | Annotations {}".format(len(lst_all_imgs), len(lst_all_annos)))
    print("There are {} bbox labels".format(cnt_bbox))

    json_string = json.dumps(attrDict, indent=2)
    with open(dst_root + dst_name, "w+") as f:
        f.write(json_string)


labelme2cocolike(json_roots=json_roots, dst_root=dst_root, dst_name=dst_name, dataset=dataset)
