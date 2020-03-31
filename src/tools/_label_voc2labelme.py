# Convert VOC .xml to labelme .json format

from collections import OrderedDict
import json
import os
import sys

from utils import parseXml, class_name_kaist, class_name_obj

XML_ROOT = ''
DST_ROOT = XML_ROOT[:-1] + '_json/'
if not os.path.exists(DST_ROOT):
    os.makedirs(DST_ROOT)
ROOT = '/workspace/tangyang.sy/Robotics_SemanticLines/data/KAIST_URBAN/'
XML_ROOTs = [
    'urban18-highway/postprocess_resexp0/',
    'urban19-highway/postprocess_resexp0/',
    'urban20-highway/postprocess_resexp0/',
    'urban21-highway/postprocess_resexp0/',
    'urban22-highway/postprocess_resexp0/',
    'urban23-highway/postprocess_resexp0/',
    'urban24-highway/postprocess_resexp0/',
    'urban25-highway/postprocess_resexp0/',
    'urban27-dongtan/postprocess_resexp0/',
    'urban29-pankyo/postprocess_resexp0/',
    'urban30-gangnam/postprocess_resexp0/',
    'urban31-gangnam/postprocess_resexp0/',
    'urban32-yeouido/postprocess_resexp0/',
    'urban33-yeouido/postprocess_resexp0/',
    'urban34-yeouido/postprocess_resexp0/',
    'urban35-seoul/postprocess_resexp0/',
    'urban36-seoul/postprocess_resexp0/',
    'urban37-seoul/postprocess_resexp0/',
]

class2skip = [
    #'sign', 'window', 'tree'
]


def voc2labelmelike(xml_root, dst_root, sub_pth=None):
    print("\n*******************CLASS TO SKIP***************************")
    print(class2skip)
    print("***********************************************************")

    xml_pth = xml_root + sub_pth if sub_pth is not None else xml_root
    if isinstance(xml_pth, list):
        xml_pth = xml_pth[0]

    lst_xml = sorted([f for f in os.listdir(xml_pth) if '.xml' in f])

    for i, xml_file in enumerate(lst_xml):
        width, height, bboxes = parseXml(xml_pth + xml_file)
        if len(bboxes) == 0 and width == 0 and height == 0:
            continue

        lst_lines = list()
        for bbox in bboxes:
            line = OrderedDict()
            if bbox[4] not in class_name_kaist and bbox[4] not in class_name_obj:
                continue
            line['label'] = bbox[4]
            direction = 1.0 if bbox[-2] == 'lt2rb' else 0.0
            if direction == 1.0:
                x_left = min(bbox[0], bbox[2])
                y_left = min(bbox[1], bbox[3])
                x_right = max(bbox[0], bbox[2])
                y_right = max(bbox[1], bbox[3])
            elif direction == 0.0:
                x_left = min(bbox[0], bbox[2])
                y_left = max(bbox[1], bbox[3])
                x_right = max(bbox[0], bbox[2])
                y_right = min(bbox[1], bbox[3])
            else:  # direction == 2.0, bbox
                x_left = bbox[0]
                y_left = bbox[1]
                x_right = bbox[2]
                y_right = bbox[3]

            line['points'] = [[x_left, y_left], [x_right, y_right]]
            line['shape_type'] = 'line' if bbox[4] in class_name_kaist else 'rectangle'
            line['line_color'] = None
            line['fill_color'] = None
            line['score'] = float(bbox[-1])
            lst_lines.append(line)

        json_dict = OrderedDict()
        json_dict['shapes'] = lst_lines
        json_dict['imagePath'] = xml_file[:-4] + '.png'
        json_dict['imageHeight'] = height
        json_dict['imageWidth'] = width
        json_dict['version'] = '3.16.2'
        json_dict['imageData'] = None
        json_dict['flags'] = {}
        json_dict['lineColor'] = [0, 255, 0, 128]
        json_dict['fillColor'] = [255, 0, 0, 128]

        dst_pth = dst_root + sub_pth[:-1] + '_labelme/' if sub_pth is not None else dst_root
        if not os.path.exists(dst_pth):
            os.makedirs(dst_pth)
        json_dst = dst_pth + xml_file[:-4] + '.json'
        with open(json_dst, 'w+') as j_out:
            json.dump(json_dict, j_out, indent=2)

        sys.stdout.write('\r>> Converting labels %d/%d' % (
            i + 1, len(lst_xml)))
        sys.stdout.flush()

    print("\nDONE")


if __name__ == '__main__':
    if XML_ROOTs is not None:
        for XML_ROOT in XML_ROOTs:
            print("Converting directory {}...".format(XML_ROOT))
            XML_ROOT = ROOT + XML_ROOT
            DST_ROOT = XML_ROOT[:-1] + '_json/'
            if not os.path.exists(DST_ROOT):
                os.makedirs(DST_ROOT)

            voc2labelmelike(xml_root=XML_ROOT, dst_root=DST_ROOT)
    else:
        voc2labelmelike(xml_root=XML_ROOT, dst_root=DST_ROOT)
