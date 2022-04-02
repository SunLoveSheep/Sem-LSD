from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import time
import torch

from models.model import create_model, load_model, create_model_torch_script
from utils.image import get_affine_transform
from utils.debugger import Debugger

from collections import OrderedDict
import os
import sys
sys.path.append("/workspace/tangyang.sy/pytorch_SemLSD_github")
from general_utils.utils import parseLineCenterNet, parseBBoxesCornerNetLite, \
    parseLineDict, parseBBoxDict, \
    writeXml, writeXml_line


def getModelParam(model, model_path):
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, ' \
                            'loaded shape{}.'.format(
                    k, model_state_dict[k].shape, state_dict[k].shape))
                state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k))
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k))
            state_dict[k] = model_state_dict[k]

    return state_dict


class BaseDetector(object):
    def __init__(self, opt):
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')
        
        print('Creating model...')
        print(opt)
        self.model = create_model(opt.arch, opt.heads, opt.head_conv)
        self.model = load_model(self.model, opt.load_model)
        self.model = self.model.to(opt.device)
        self.model.eval()

        self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
        self.max_per_image = 100
        self.num_classes = opt.num_classes
        self.scales = opt.test_scales
        self.opt = opt
        self.pause = True
        self.image_counter = 0

    def pre_process(self, image, scale, meta=None):
        height, width = image.shape[0:2]
        new_height = int(height * scale)
        new_width  = int(width * scale)
        if self.opt.fix_res:
            inp_height, inp_width = self.opt.input_h, self.opt.input_w
            c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
            s = max(height, width) * 1.0
        else:
            inp_height = (new_height | self.opt.pad) + 1
            inp_width = (new_width | self.opt.pad) + 1
            c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
            s = np.array([inp_width, inp_height], dtype=np.float32)

        trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
        resized_image = cv2.resize(image, (new_width, new_height))
        inp_image = cv2.warpAffine(
            resized_image, trans_input, (inp_width, inp_height),
            flags=cv2.INTER_LINEAR)

        # Test resize.
        # image_padding = np.zeros((120, 640, 3), dtype=np.uint8)
        # image_data_padding = np.vstack([image_padding, image, image_padding])
        # inp_image = cv2.resize(image_data_padding, (512, 512))

        inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)
        images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)

        if self.opt.flip_test:
            images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
        images = torch.from_numpy(images)
        meta = {'c': c, 's': s, 
                        'out_height': inp_height // self.opt.down_ratio, 
                        'out_width': inp_width // self.opt.down_ratio}
        return images, meta

    def pre_process_line_att(self, line_att):
        height, width = line_att.shape[0:2]
        new_height = 128
        new_width = 128
        inp_height, inp_width = new_height, new_width
        c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
        s = max(height, width) * 1.0

        trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
        resized_line_att = cv2.resize(line_att, (new_width, new_height))
        inp_line_att = cv2.warpAffine(
            resized_line_att, trans_input, (inp_width, inp_height),
            flags=cv2.INTER_LINEAR)
        inp_line_att = (inp_line_att.astype(np.float32) / 255.)

        # Test resize.
        # image_padding = np.zeros((120, 640, 3), dtype=np.uint8)
        # image_data_padding = np.vstack([image_padding, image, image_padding])
        # inp_image = cv2.resize(image_data_padding, (512, 512))

        inp_line_att = torch.from_numpy(inp_line_att)

        if self.opt.fp16:
            inp_line_att = inp_line_att.half()
        return inp_line_att

    def process(self, images, return_time=False):
        raise NotImplementedError

    def process_temporal(self, images, return_time=False, ft_1=None, ft_2=None):
        raise NotImplementedError

    def post_process(self, dets, meta, scale=1):
        raise NotImplementedError

    def merge_outputs(self, detections):
        raise NotImplementedError

    def debug(self, debugger, images, dets, output, scale=1):
        raise NotImplementedError

    def show_results(self, debugger, image, results, cnt, img_name):
     raise NotImplementedError

    def run(self, image_or_path_or_tensor, meta=None):
        load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
        merge_time, tot_time = 0, 0
        debugger = Debugger(dataset=self.opt.dataset, ipynb=(self.opt.debug==3),
                            theme=self.opt.debugger_theme)
        start_time = time.time()
        pre_processed = False
        if isinstance(image_or_path_or_tensor, np.ndarray):
            image = image_or_path_or_tensor
        elif type(image_or_path_or_tensor) == type (''): 
            image = cv2.imread(image_or_path_or_tensor)
        else:
            image = image_or_path_or_tensor['image'][0].numpy()
            pre_processed_images = image_or_path_or_tensor
            pre_processed = True

        try:
            _, _, _ = image.shape
        except AttributeError:
            print("Nonetype image at {}".format(image_or_path_or_tensor))

        loaded_time = time.time()
        load_time += (loaded_time - start_time)
        
        detections = []
        for scale in self.scales:
            scale_start_time = time.time()
            if not pre_processed:
                images, meta = self.pre_process(image, scale, meta)
            else:
                # import pdb; pdb.set_trace()
                images = pre_processed_images['images'][scale][0]
                meta = pre_processed_images['meta'][scale]
                meta = {k: v.numpy()[0] for k, v in meta.items()}
            images = images.to(self.opt.device)
            if 'cpu' not in self.opt.device.type:
                torch.cuda.synchronize()
            pre_process_time = time.time()
            pre_time += pre_process_time - scale_start_time

            output, dets, forward_time = self.process(images, return_time=True)

            if 'cpu' not in self.opt.device.type:
                torch.cuda.synchronize()
            net_time += forward_time - pre_process_time
            decode_time = time.time()
            dec_time += decode_time - forward_time
            
            if self.opt.debug >= 2:
                self.debug(debugger, images, dets, output, scale)
            
            dets = self.post_process(dets, meta, scale)
            if 'cpu' not in self.opt.device.type:
                torch.cuda.synchronize()
            post_process_time = time.time()
            post_time += post_process_time - decode_time

            detections.append(dets)
        
        results = self.merge_outputs(detections)
        if 'cpu' not in self.opt.device.type:
            torch.cuda.synchronize()
        end_time = time.time()
        merge_time += end_time - post_process_time
        tot_time += end_time - start_time

        # parse to our general format:
        cate_dict = OrderedDict()
        num_classes = 80
        if self.opt.dataset == 'semantic_line_kaist':
            num_classes = 14
        for i in range(num_classes):
            cate_dict.update({str(i): debugger.names[i]})
        xml_pth = self.opt.save_path.replace("Images", "Preds")
        if not os.path.exists(xml_pth):
            os.makedirs(xml_pth)
        if self.opt.task == 'ctdet_line':
            bboxesToxml = parseLineCenterNet(results, image)
            bboxes_dict_to_xml = parseLineDict(bboxesToxml['detection_boxes'],
                                               bboxesToxml['detection_classes'],
                                               bboxesToxml['detection_scores'],
                                               bboxesToxml['detection_directs'],
                                               cate_dict,
                                               min_score_thresh=self.opt.vis_thresh)
            writeXml_line(box_dict_in=bboxes_dict_to_xml,
                          image_filename=image_or_path_or_tensor,
                          image_in=image,
                          image_dir=image_or_path_or_tensor.split('/')[-2],
                          image_dst_in=self.opt.save_path,
                          xml_dst_in=xml_pth)
        else:
            bboxesToxml = parseBBoxesCornerNetLite(results, image)
            bboxes_dict_to_xml = parseBBoxDict(bboxesToxml['detection_boxes'],
                                               bboxesToxml['detection_classes'],
                                               bboxesToxml['detection_scores'],
                                               cate_dict,
                                               min_score_thresh=self.opt.vis_thresh)
            writeXml(box_dict_in=bboxes_dict_to_xml,
                     image_filename=image_or_path_or_tensor,
                     image_in=image,
                     image_dir=image_or_path_or_tensor.split('/')[-2],
                     image_dst_in=self.opt.save_path,
                     xml_dst_in=xml_pth)

        if self.opt.debug >= 1:
            self.show_results(debugger, image, results, self.image_counter, image_or_path_or_tensor)
        self.image_counter += 1
        return {'results': results, 'tot': tot_time, 'load': load_time,
                        'pre': pre_time, 'net': net_time, 'dec': dec_time,
                        'post': post_time, 'merge': merge_time}
