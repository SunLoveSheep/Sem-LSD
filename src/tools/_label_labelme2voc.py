# Convert .json from labelme to VOC .xml format.

from collections import OrderedDict
import json
import os

from utils import *

class2skip = [
    # 'sign', 'window', 'tree'
]

# ROOT = '/Users/yisun/Desktop/Robotics_SemanticLines/data/KAIST_labeled/'
# ROOT = '/Users/yisun/Desktop/Robotics_SemanticLines/data/KAIST_q28-db39/kaist_triplet_q28/201-400/'
ROOT = '/Users/yisun/Desktop/Robotics_SemanticLines/data/KAIST_URBAN/pilot_semlsd_eval_18seqs/urban29-pankyo/'
JSON_ROOT = ROOT + 'rectified_image/'
# XML_DST = ROOT + 'KAIST_combined_0823_xml/'
# ROOT = '/Users/yisun/Desktop/Robotics_SemanticLines/data/TZKJ_HDMap/'
# ROOT = '/Users/yisun/Desktop/python/pytorch_ComputerVision/test_imgs/wyxl_1705-1714_cv2/output_line/'
# ROOT = '/workspace/tangyang.sy/pytorch_CV/test_imgs/WYXL_1705-1714/'
# ROOT = '/Users/yisun/Desktop/Robotics_SemanticLines/data/KITTI_labeled/1w5_images/'
# JSON_ROOT = ROOT + 'images/val_semantic_line/'
# JSON_ROOT = ROOT + '81506_test/'
XML_DST = JSON_ROOT[:-1] + '_xml/'


def labelme2voclike(json_root, dst_root, sub_pth=None):
    print("\n*******************CLASS TO SKIP***************************")
    print(class2skip)
    print("***********************************************************")

    json_pth = json_root + sub_pth if sub_pth is not None else json_root
    if isinstance(json_root, list):
        json_root = json_root[0]
    if isinstance(json_pth, list):
        json_pth = json_pth[0]

    lst_json = [f for f in os.listdir(json_pth) if '.json' in f]

    for json_file in lst_json:
        with open(json_pth + '/' + json_file, 'r') as j_in:
            json_dict = OrderedDict(json.load(j_in))

        root = ET.Element("annotation")
        folder = ET.SubElement(root, "folder")
        folder.text = json_root.split('/')[-1] if json_root[-1] != '/' else json_root.split('/')[-2]
        filename = ET.SubElement(root, "filename")
        filename.text = folder.text + '_' + json_dict['imagePath']

        path = ET.SubElement(root, "path")
        path.text = json_root + sub_pth if sub_pth is not None else json_root

        try:
            h, w, c = json_dict['imageHeight'], json_dict['imageWidth'], 3
        except KeyError:
            print("imageHeight and imageWidth not provided in .json file!! Use KAIST default 1280x560x3")
            h, w, c = 560, 1280, 3

        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(w)
        ET.SubElement(size, "height").text = str(h)
        ET.SubElement(size, "depth").text = str(c)

        segmented = ET.SubElement(root, "segmented")
        segmented.text = "0"

        for shape in json_dict['shapes']:
            shape['label'] = shape['label'].replace('-', '_')  # replace '-' (if any) with '_'
            shape['label'] = shape['label'].replace('glass', 'grass')  # replace 'glass' (if any) with 'grass'
            shape['label'] = shape['label'].replace('building\\', 'building')  # replace 'glass' (if any) with 'grass'

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

            for pts in lst_pt_pairs:
                object_box = ET.SubElement(root, "object")
                name = fix_typos(shape['label'])
                ET.SubElement(object_box, "name").text = name
                try:
                    ET.SubElement(object_box, "score").text = str(shape['score'])
                except KeyError:
                    ET.SubElement(object_box, "score").text = str(1.0)

                ET.SubElement(object_box, "pose").text = "Unspecified"
                ET.SubElement(object_box, "truncated").text = "0"
                ET.SubElement(object_box, "difficult").text = "0"

                if pts[0][0] < pts[1][0]:
                    x_left = pts[0][0]
                    y_left = pts[0][1]
                    x_right = pts[1][0]
                    y_right = pts[1][1]
                else:
                    x_left = pts[1][0]
                    y_left = pts[1][1]
                    x_right = pts[0][0]
                    y_right = pts[0][1]
                if y_left < y_right:
                    direction = 'lt2rb'
                else:
                    direction = 'lb2rt'

                bndbox = ET.SubElement(object_box, "bndbox")
                ET.SubElement(bndbox, "xmin").text = str(x_left)
                ET.SubElement(bndbox, "ymin").text = str(y_left)
                ET.SubElement(bndbox, "xmax").text = str(x_right)
                ET.SubElement(bndbox, "ymax").text = str(y_right)
                ET.SubElement(object_box, "direction").text = direction

        xmlstr = prettify(root)

        dst_pth = dst_root + sub_pth[:-1] + '_voc/' if sub_pth is not None else dst_root
        if not os.path.exists(dst_pth):
            os.makedirs(dst_pth)
        xml_dst = dst_pth + json_file[:-5] + '.xml'
        with open(xml_dst, 'w+') as f:
            f.write(xmlstr)

    print("DONE")


labelme2voclike(json_root=JSON_ROOT, dst_root=XML_DST)