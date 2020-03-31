# -*- coding:utf-8 -*-
# !/usr/bin/env python

'''
https://github.com/fengzhongyouxia/TensorExpand/blob/master/TensorExpand/Object%20detection/Data_interface/MSCOCO/Pascal%20VOC/PascalVOC2COCO.py
'''

import json
import cv2
import numpy as np
import glob
import PIL.Image
import os, sys
import xml.etree.ElementTree as ET
from utils import parseXml


class PascalVOC2coco(object):
    def __init__(self, xml=[], save_json_path='./new.json', phase='train'):
        '''
        :param xml: 所有Pascal VOC的xml文件路径组成的列表
        :param save_json_path: json保存位置
        '''
        self.xml = xml
        self.save_json_path = save_json_path
        self.images = []
        self.categories = []
        self.annotations = []
        # self.data_coco = {}
        self.label = []
        self.annID = 1
        self.height = 0
        self.width = 0
        self.phase = phase

        self.save_json()

    def data_transfer(self):
        for num, json_file in enumerate(self.xml):

            # 进度输出
            sys.stdout.write('\r>> Converting image %d/%d' % (
                num + 1, len(self.xml)))
            sys.stdout.flush()

            self.json_file = json_file
            self.num = num
            
            width, height, bboxes = parseXml(json_file)
            if len(bboxes) == 0 and width == 0 and height == 0:
                continue
            path = self.json_file.replace('annotations/{}_xml/'.format(phase), 'images/{}_imgs/'.format(phase))
            path = os.path.dirname(path)
            # print("path: ", path)
            obj_path = glob.glob(os.path.join(path, '*.png'))

            # self.file_name = self.json_file.split('/')[-1][:-4]
            self.file_name = self.json_file.split('/')[-1][:-4]
            self.path = self.file_name + '.png'
            self.file_name = self.file_name + '.png' if '.png' not in self.file_name and '.jpg' not in self.file_name \
                else self.file_name
            self.width = width
            self.height = height
            self.images.append(self.image())

            for bbox in bboxes:
                bbox[4] = bbox[4].replace('Cyclist', 'cyclist')
                if bbox[4] != 'person' and bbox[4] != 'car' and bbox[4] != 'cyclist':
                    continue
                self.supercategory = bbox[4]
                if self.supercategory not in self.label:
                    self.categories.append(self.categorie())
                    self.label.append(self.supercategory)

                xmin = bbox[0]
                ymin = bbox[1]
                xmax = bbox[2]
                ymax = bbox[3]
                self.rectangle = [xmin, ymin, xmax, ymax]
                self.bbox = [xmin, ymin, abs(xmax - xmin), abs(ymax - ymin)]  # COCO 对应格式[x,y,w,h]
                
                self.score = bbox[-1]
                self.annotations.append(self.annotation())
                self.annID += 1

        sys.stdout.write('\n')
        sys.stdout.flush()

    def image(self):
        image = {}
        image['height'] = self.height
        image['width'] = self.width
        image['id'] = self.num + 1
        image['file_name'] = self.file_name
        return image

    def categorie(self):
        categorie = {}
        categorie['supercategory'] = self.supercategory
        categorie['id'] = len(self.label) + 1  # 0 默认为背景
        categorie['name'] = self.supercategory
        return categorie

    @staticmethod
    def change_format(contour):
        contour2 = []
        length = len(contour)
        for i in range(0, length, 2):
            contour2.append([contour[i], contour[i + 1]])
        return np.asarray(contour2, np.int32)

    def annotation(self):
        annotation = {}
        # annotation['segmentation'] = [self.getsegmentation()]
        annotation['segmentation'] = [list(map(float, self.getsegmentation()))]
        annotation['iscrowd'] = 0
        annotation['image_id'] = self.num + 1
        # annotation['bbox'] = list(map(float, self.bbox))
        annotation['bbox'] = self.bbox
        annotation['category_id'] = self.getcatid(self.supercategory)
        annotation['id'] = self.annID
        annotation['score'] = float(self.score)

        # 计算轮廓面积
        # contour = PascalVOC2coco.change_format(annotation['segmentation'][0])
        # annotation['area'] = abs(cv2.contourArea(contour, True))
        annotation['area'] = self.bbox[2] * self.bbox[3]

        return annotation

    def getcatid(self, label):
        for categorie in self.categories:
            if label == categorie['name']:
                return categorie['id']
        return -1

    def getsegmentation(self):

        try:
            mask_1 = cv2.imread(self.path, 0)
            mask = np.zeros_like(mask_1, np.uint8)
            rectangle = self.rectangle
            mask[rectangle[1]:rectangle[3], rectangle[0]:rectangle[2]] = mask_1[rectangle[1]:rectangle[3],
                                                                         rectangle[0]:rectangle[2]]

            # 计算矩形中点像素值
            mean_x = (rectangle[0] + rectangle[2]) // 2
            mean_y = (rectangle[1] + rectangle[3]) // 2

            end = min((mask.shape[1], int(rectangle[2]) + 1))
            start = max((0, int(rectangle[0]) - 1))

            flag = True
            for i in range(mean_x, end):
                x_ = i
                y_ = mean_y
                pixels = mask_1[y_, x_]
                if pixels != 0 and pixels != 220:  # 0 对应背景 220对应边界线
                    mask = (mask == pixels).astype(np.uint8)
                    flag = False
                    break
            if flag:
                for i in range(mean_x, start, -1):
                    x_ = i
                    y_ = mean_y
                    pixels = mask_1[y_, x_]
                    if pixels != 0 and pixels != 220:
                        mask = (mask == pixels).astype(np.uint8)
                        break
            self.mask = mask

            return self.mask2polygons()

        except:
            return [0]

    def mask2polygons(self):
        '''从mask提取边界点'''
        contours = cv2.findContours(self.mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找到轮廓线
        bbox = []
        for cont in contours[1]:
            [bbox.append(i) for i in list(cont.flatten())]
            # map(bbox.append,list(cont.flatten()))
        return bbox  # list(contours[1][0].flatten())

    # '''
    def getbbox(self, points):
        '''边界点生成mask，从mask提取定位框'''
        # img = np.zeros([self.height,self.width],np.uint8)
        # cv2.polylines(img, [np.asarray(points)], True, 1, lineType=cv2.LINE_AA)  # 画边界线
        # cv2.fillPoly(img, [np.asarray(points)], 1)  # 画多边形 内部像素值为1
        polygons = points
        mask = self.polygons_to_mask([self.height, self.width], polygons)
        return self.mask2box(mask)

    def mask2box(self, mask):
        '''从mask反算出其边框
        mask：[h,w]  0、1组成的图片
        1对应对象，只需计算1对应的行列号（左上角行列号，右下角行列号，就可以算出其边框）
        '''
        # np.where(mask==1)
        index = np.argwhere(mask == 1)
        rows = index[:, 0]
        clos = index[:, 1]
        # 解析左上角行列号
        left_top_r = np.min(rows)  # y
        left_top_c = np.min(clos)  # x

        # 解析右下角行列号
        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)

        # return [(left_top_r,left_top_c),(right_bottom_r,right_bottom_c)]
        # return [(left_top_c, left_top_r), (right_bottom_c, right_bottom_r)]
        # return [left_top_c, left_top_r, right_bottom_c, right_bottom_r]  # [x1,y1,x2,y2]
        return [left_top_c, left_top_r, right_bottom_c - left_top_c,
                right_bottom_r - left_top_r]  # [x1,y1,w,h] 对应COCO的bbox格式

    def polygons_to_mask(self, img_shape, polygons):
        '''边界点生成mask'''
        mask = np.zeros(img_shape, dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        return mask

    # '''
    def data2coco(self):
        data_coco = {}
        data_coco['images'] = self.images
        data_coco['categories'] = self.categories
        data_coco['annotations'] = self.annotations
        return data_coco

    def save_json(self):
        self.data_transfer()
        self.data_coco = self.data2coco()
        # 保存json文件
        json.dump(self.data_coco, open(self.save_json_path, 'w'), indent=4)  # indent=4 更加美观显示


phase = 'train'
# xml_root = '/workspace/tangyang.sy/datasets_obj_detect/voc2coco/combined_v2_20190318_cont_9bags_coco/' \
#           'annotations/{}_xml/'.format(phase)
# xml_root = '/Users/yisun/Desktop/Robotics_LabelUtils/combined_xixiyq_test/annotations/{}_xml/'.format(phase)
# xml_root = '/workspace/tangyang.sy/datasets/HCY/20190103_combined_ctdet_xixiyq_resdsod18_deconv2x2_cut4_512_ep170/'
# xml_root = '/Users/yisun/Desktop/data/ObjDetection/XIXIYQ/Annotations/'
xml_root = '/Users/yisun/Desktop/Robotics_SemanticLines/data/KAIST_URBAN/' \
           '20200115_zhongbao/images/{}_semantic_line/'.format(phase)
# xml_lst = '/Users/yisun/Desktop/data/ObjDetection/XIXIYQ/cus_{}.txt'.format(phase)
xml_lst = None
if xml_lst is not None:
    with open(xml_lst, 'r') as lst_in:
        lines = lst_in.readlines()
        xml_files = [xml_root + f[:-1] + '.xml' for f in lines]
else:
    xml_files = glob.glob(xml_root + '*.xml'.format(phase))
# xml_file = [xml_root + l for l in os.listdir(xml_root)]
# json_file = xml_root + '../{}_9bags_wKITTI.json'.format(phase)
# json_file = xml_root + '../20190103_combined_ctdet_xixiyq_resdsod18_deconv2x2_cut4_512_ep170.json'
json_file = xml_root + '../../annotations/{}_semantic_line.json'.format(phase)

PascalVOC2coco(xml_files, json_file, phase)
