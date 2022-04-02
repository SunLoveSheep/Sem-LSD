# utility functions used for customized training / testing / etc.
from collections import OrderedDict
import xml.etree.ElementTree as ET
from xml.dom import minidom
import json
import os

CATEDICT_COCO = '/workspace/tangyang.sy/pytorch_CV/general_utils/COCO_CateDict.json'


def loadCateDictCOCO():
    if not os.path.isfile(CATEDICT_COCO):
        print("COCODICT_COCO not created yet! Please run 'writeCateDictCOCO' function first with"
              "one detection result from COCO-trained model (currently support CornerNet-Lite).")
        return None

    with open(CATEDICT_COCO, 'r') as d_in:
        cate_dict = OrderedDict(json.load(d_in))
    # cate_dict = sorted(cate_dict.items())
    return cate_dict


# run once: from a cornernet-lite detection results to get category list
def writeCateDictCOCO(bboxes_dict_in):
    cate_dict = dict()

    for i, (cate, _) in enumerate(sorted(bboxes_dict_in.items())):
        cate_dict.update({i: cate})

    with open(CATEDICT_COCO, 'w+') as d_out:
        json.dump(cate_dict, d_out)


# get CateDict COCO:
def genCateDictCOCO(bboxes_dict_in):
    cate_dict = dict()
    for i, (cate, _) in enumerate(sorted(bboxes_dict_in.items())):
        cate_dict.update({i: cate})

    return cate_dict


def parseBBoxesCornerNetLite(bboxes_dict_in, image_in):
    h, w, _ = image_in.shape

    bboxes_dict_out = OrderedDict({
        'num_detections': 0,
        'detection_boxes': list(),
        'detection_scores': list(),
        'detection_classes': list()
    })

    lst_boxes = list()
    lst_scores = list()
    lst_classes = list()
    for i, (cate, bboxes) in enumerate(sorted(bboxes_dict_in.items())):
        # Skip categories with empty detections
        if len(bboxes) == 0:
            continue

        for bbox in bboxes:
            xmin = bbox[0] / float(w)
            ymin = bbox[1] / float(h)
            xmax = bbox[2] / float(w)
            ymax = bbox[3] / float(h)
            score = bbox[4]

            lst_boxes.append([xmin, ymin, xmax, ymax])
            lst_scores.append(score)
            lst_classes.append(i)

        bboxes_dict_out['num_detections'] += len(bboxes)

    # sorted by scores:
    comb = zip(lst_scores,
                lst_boxes,
                lst_classes)

    lst_scores, \
    lst_boxes, \
    lst_classes = zip(*sorted(comb, reverse=True))
    bboxes_dict_out['detection_scores'] = list(lst_scores)
    bboxes_dict_out['detection_boxes'] = list(lst_boxes)
    bboxes_dict_out['detection_classes'] = list(lst_classes)

    return bboxes_dict_out


def parseLineCenterNet(bboxes_dict_in, image_in):
    h, w, _ = image_in.shape

    bboxes_dict_out = OrderedDict({
        'num_detections': 0,
        'detection_boxes': list(),
        'detection_scores': list(),
        'detection_classes': list()
    })

    lst_boxes = list()
    lst_scores = list()
    lst_directs = list()
    lst_classes = list()
    for i, (cate, bboxes) in enumerate(sorted(bboxes_dict_in.items())):
        # Skip categories with empty detections
        if len(bboxes) == 0:
            continue

        for bbox in bboxes:
            xmin = bbox[0] / float(w)
            ymin = bbox[1] / float(h)
            xmax = bbox[2] / float(w)
            ymax = bbox[3] / float(h)
            score = bbox[4]
            direct = bbox[5]

            lst_boxes.append([xmin, ymin, xmax, ymax])
            lst_scores.append(score)
            lst_directs.append(direct)
            lst_classes.append(i)

        bboxes_dict_out['num_detections'] += len(bboxes)

    # sorted by scores:
    comb = zip(lst_scores,
               lst_boxes,
               lst_directs,
               lst_classes)

    lst_scores, \
    lst_boxes, \
    lst_directs, \
    lst_classes = zip(*sorted(comb, reverse=True))
    bboxes_dict_out['detection_scores'] = list(lst_scores)
    bboxes_dict_out['detection_directs'] = list(lst_directs)
    bboxes_dict_out['detection_boxes'] = list(lst_boxes)
    bboxes_dict_out['detection_classes'] = list(lst_classes)

    return bboxes_dict_out


def convertCOCOLabelM2Det(label_in):
    label_dict = OrderedDict()

    for l in range(len(label_in)):
        label_dict.update({l: label_in[l]})

    return label_dict


def parseBBoxesM2Det(bboxes_in, image_in):
    bboxes_dict_out = OrderedDict({
        'num_detections': 0,
        'detection_boxes': list(),
        'detection_scores': list(),
        'detection_classes': list()
    })

    try:
        boxes = bboxes_in[:, :4]
    except IndexError:
        print(bboxes_in)
        return bboxes_dict_out
    scores = bboxes_in[:, 4]
    cls_inds = bboxes_in[:, 5]

    h, w, _ = image_in.shape

    lst_boxes = list()
    lst_scores = list()
    lst_classes = list()
    for b in range(boxes.shape[0]):
        xmin = boxes[b][0] / float(w)
        ymin = boxes[b][1] / float(h)
        xmax = boxes[b][2] / float(w)
        ymax = boxes[b][3] / float(h)

        lst_boxes.append([xmin, ymin, xmax, ymax])
        lst_scores.append(scores[b])
        lst_classes.append(int(cls_inds[b]))

    bboxes_dict_out['num_detections'] += boxes.shape[0]

    # sorted by scores:
    comb = zip(lst_scores,
               lst_boxes,
               lst_classes)

    lst_scores, \
    lst_boxes, \
    lst_classes = zip(*sorted(comb, reverse=True))
    bboxes_dict_out['detection_scores'] = list(lst_scores)
    bboxes_dict_out['detection_boxes'] = list(lst_boxes)
    bboxes_dict_out['detection_classes'] = list(lst_classes)

    return bboxes_dict_out


# get box_dict to feed to the function for writing to .xml files.
def parseBBoxDict(
        boxes,
        classes,
        scores,
        category_index=None,
        min_score_thresh=0.1
):
    boxes_dict = OrderedDict()
    max_num = len(boxes)
    for i in range(max_num):
        if scores is None or scores[i] > min_score_thresh:
            box = tuple(boxes[i])
            if classes[i] is not str:
                classes[i] = str(classes[i])
            if classes[i] in category_index.keys():
                class_name = category_index[classes[i]]
            else:
                class_name = 'N/A'
            display_str = str(class_name)
            # if not display_str:
            #    display_str = '{}%'.format(int(100 * scores[i]))
            # else:
            #    display_str = '{}: {}%'.format(display_str, int(100 * scores[i]))
            boxes_dict[box] = [display_str, scores[i]]

    return boxes_dict


# get box_dict to feed to the function for writing to .xml files.
def parseLineDict(
        boxes,
        classes,
        scores,
        directs,
        category_index=None,
        min_score_thresh=0.1
):
    boxes_dict = OrderedDict()
    max_num = len(boxes)
    for i in range(max_num):
        if classes[i] is not str:
            classes[i] = str(classes[i])
        class_name = category_index[classes[i]]
        thres = min_score_thresh
        if scores is None or scores[i] > thres:
            box = tuple(boxes[i])
            if classes[i] in category_index.keys():
                class_name = category_index[classes[i]]
            else:
                class_name = 'N/A'
            display_str = str(class_name)
            # if not display_str:
            #    display_str = '{}%'.format(int(100 * scores[i]))
            # else:
            #    display_str = '{}: {}%'.format(display_str, int(100 * scores[i]))
            boxes_dict[box] = [display_str, scores[i], directs[i]]

    return boxes_dict


# prettify the output format of xml
def _prettify(elem, doctype = None):
    """Return a pretty-printed XML string for the Element.
        """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    if doctype is not None:
        reparsed.insertBefore(doctype, reparsed.documentElement)
    return reparsed.toprettyxml(indent="  ")


# write detection dictionary to .xml to support future human labeling
def writeXml(box_dict_in, image_filename, image_in, image_dir, image_dst_in, xml_dst_in):
    root = ET.Element("annotation")
    folder = ET.SubElement(root, "folder")
    folder.text = image_dir.split('/')[-1]
    filename = ET.SubElement(root, "filename")
    filename.text = image_filename.split('/')[-1]

    path = ET.SubElement(root, "path")
    path.text = image_dst_in

    h, w, c = image_in.shape

    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(w)
    ET.SubElement(size, "height").text = str(h)
    ET.SubElement(size, "depth").text = str(c)

    segmented = ET.SubElement(root, "segmented")
    segmented.text = "0"

    # bounding boxes:
    for box, str_score in box_dict_in.items():
        object_box = ET.SubElement(root, "object")
        display_str = str_score[0]
        display_score = str_score[1]
        display_str = 'person' if display_str == 'pedestrian' else display_str
        ET.SubElement(object_box, "name").text = display_str
        ET.SubElement(object_box, "score").text = str(display_score)

        ET.SubElement(object_box, "pose").text = "Unspecified"
        ET.SubElement(object_box, "truncated").text = "0"
        ET.SubElement(object_box, "difficult").text = "0"

        bndbox = ET.SubElement(object_box, "bndbox")
        xmin = box[0]
        ymin = box[1]
        xmax = box[2]
        ymax = box[3]
        ymin = int(ymin * h)
        xmin = int(xmin * w)
        ymax = int(ymax * h)
        xmax = int(xmax * w)
        ET.SubElement(bndbox, "xmin").text = str(xmin)
        ET.SubElement(bndbox, "ymin").text = str(ymin)
        ET.SubElement(bndbox, "xmax").text = str(xmax)
        ET.SubElement(bndbox, "ymax").text = str(ymax)

    xmlstr = _prettify(root)

    xml_dst = xml_dst_in + image_filename.split('/')[-1][:-4] + '.xml'
    with open(xml_dst, "w+") as f:
        f.write(xmlstr)


# write detection dictionary to .xml to support future human labeling
def writeXml_line(box_dict_in, image_filename, image_in, image_dir, image_dst_in, xml_dst_in):
    root = ET.Element("annotation")
    folder = ET.SubElement(root, "folder")
    folder.text = image_dir.split('/')[-1]
    filename = ET.SubElement(root, "filename")
    filename.text = image_filename.split('/')[-1]

    path = ET.SubElement(root, "path")
    path.text = image_dst_in

    h, w, c = image_in.shape

    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(w)
    ET.SubElement(size, "height").text = str(h)
    ET.SubElement(size, "depth").text = str(c)

    segmented = ET.SubElement(root, "segmented")
    segmented.text = "0"

    # bounding boxes:
    for box, str_score in box_dict_in.items():
        object_box = ET.SubElement(root, "object")
        display_str = str_score[0]
        display_score = str_score[1]
        if str_score[2] == 1.0:
            display_direct = 'lt2rb'
        elif str_score[2] == 0.0:
            display_direct = 'lb2rt'
        else:
            display_direct = 'bbox'
        # display_direct = 'lt2rb' if str_score[2] == 1.0 else 'lb2rt'
        display_str = 'person' if display_str == 'pedestrian' else display_str
        ET.SubElement(object_box, "name").text = display_str
        ET.SubElement(object_box, "score").text = str(display_score)
        ET.SubElement(object_box, "direction").text = display_direct

        ET.SubElement(object_box, "pose").text = "Unspecified"
        ET.SubElement(object_box, "truncated").text = "0"
        ET.SubElement(object_box, "difficult").text = "0"

        bndbox = ET.SubElement(object_box, "bndbox")
        xmin = box[0]
        ymin = box[1]
        xmax = box[2]
        ymax = box[3]
        ymin = int(ymin * h)
        xmin = int(xmin * w)
        ymax = int(ymax * h)
        xmax = int(xmax * w)
        ET.SubElement(bndbox, "xmin").text = str(xmin)
        ET.SubElement(bndbox, "ymin").text = str(ymin)
        ET.SubElement(bndbox, "xmax").text = str(xmax)
        ET.SubElement(bndbox, "ymax").text = str(ymax)

    xmlstr = _prettify(root)

    xml_dst = xml_dst_in + image_filename.split('/')[-1][:-4] + '.xml'
    with open(xml_dst, "w+") as f:
        f.write(xmlstr)


# write detection dictionary to .xml to support future human labeling
def writeXmlLine(box_dict_in, image_filename, image_in, image_dir, image_dst_in, xml_dst_in):
    root = ET.Element("annotation")
    folder = ET.SubElement(root, "folder")
    folder.text = image_dir.split('/')[-1]
    filename = ET.SubElement(root, "filename")
    filename.text = image_filename.split('/')[-1]

    path = ET.SubElement(root, "path")
    path.text = image_dst_in

    h, w, c = image_in.shape

    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(w)
    ET.SubElement(size, "height").text = str(h)
    ET.SubElement(size, "depth").text = str(c)

    segmented = ET.SubElement(root, "segmented")
    segmented.text = "0"

    # bounding boxes:
    for box, str_score in box_dict_in.items():
        object_box = ET.SubElement(root, "object")
        display_str = str_score[0]
        display_score = str_score[1]
        display_str = 'person' if display_str == 'pedestrian' else display_str
        ET.SubElement(object_box, "name").text = display_str
        ET.SubElement(object_box, "score").text = str(display_score)


        ET.SubElement(object_box, "pose").text = "Unspecified"
        ET.SubElement(object_box, "truncated").text = "0"
        ET.SubElement(object_box, "difficult").text = "0"

        bndbox = ET.SubElement(object_box, "bndbox")
        xmin = box[0]
        ymin = box[1]
        xmax = box[2]
        ymax = box[3]
        ymin = int(ymin * h)
        xmin = int(xmin * w)
        ymax = int(ymax * h)
        xmax = int(xmax * w)
        ET.SubElement(bndbox, "xmin").text = str(xmin)
        ET.SubElement(bndbox, "ymin").text = str(ymin)
        ET.SubElement(bndbox, "xmax").text = str(xmax)
        ET.SubElement(bndbox, "ymax").text = str(ymax)

    xmlstr = _prettify(root)

    xml_dst = xml_dst_in + image_filename.split('/')[-1][:-4] + '.xml'
    with open(xml_dst, "w+") as f:
        f.write(xmlstr)
