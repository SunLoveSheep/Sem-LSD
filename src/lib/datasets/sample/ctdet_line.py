from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import cv2
import os
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian
import math


class CTDetLineDataset(data.Dataset):
    seq = None

    def _semline_cord_to_box(self, cord_in):
        cord = np.array([cord_in[0], cord_in[1], cord_in[2], cord_in[3]],
                                        dtype=np.float32)
        return cord

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
                i *= 2
        return border // i

    def __getitem__(self, index):
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']

        # to fit for any given global data path:
        if 'images' not in self.img_dir:
            img_folder = self.img_dir.split('/')[-1]
            self.img_dir = self.img_dir.replace(img_folder, "images/" + img_folder)

        img_path = os.path.join(self.img_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)

        img = cv2.imread(img_path)
        try:
            height, width = img.shape[0], img.shape[1]
        except AttributeError:
            print("None type image! path: {}".format(img_path))

        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        if self.opt.keep_res:
            input_h = (height | self.opt.pad) + 1
            input_w = (width | self.opt.pad) + 1
            s = np.array([input_w, input_h], dtype=np.float32)
        else:
            s = max(img.shape[0], img.shape[1]) * 1.0
            input_h, input_w = self.opt.input_h, self.opt.input_w
        
        flipped = False
        if self.split == 'train':
            if not self.opt.not_rand_crop:
                s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
                w_border = self._get_border(128, img.shape[1])
                h_border = self._get_border(128, img.shape[0])
                c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
                c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
            else:
                sf = self.opt.scale
                cf = self.opt.shift
                c[0] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
                c[1] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
                s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)

            if np.random.random() < self.opt.flip:
                flipped = True
                try:
                    img = img[:, ::-1, :]
                except IndexError:
                    img = img[:, ::-1]
                c[0] = width - c[0] - 1

        trans_input = get_affine_transform(
            c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)
        inp = (inp.astype(np.float32) / 255.)
        if self.split == 'train' and not self.opt.no_color_aug:
            color_aug(self._data_rng, inp, self._eig_val, self._eig_vec, self.opt.color_aug_var)
        inp = (inp - np.mean(self.mean)) / np.mean(self.std)
        inp = inp.transpose(2, 0, 1)

        output_h = input_h // self.opt.down_ratio
        output_w = input_w // self.opt.down_ratio
        num_classes = self.num_classes
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

        # Get ground truth gradient magnitude
        if self.opt.loss_hm_magnitude:
            img_mag_path = os.path.join(self.img_dir + '_mag', file_name.replace('.png', '_mag.png'))
            inp_grad_magnitude = cv2.imread(img_mag_path, 0)
            inp_grad_magnitude = cv2.warpAffine(inp_grad_magnitude, trans_output,
                                                (output_w, output_h),
                                                flags=cv2.INTER_LINEAR)
            inp_grad_magnitude = (inp_grad_magnitude.astype(np.float32) / 255.)

        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        direct = np.zeros((self.max_objs, 1), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        
        gt_line = []
        for k in range(num_objs):
            ann = anns[k]
            cord = self._semline_cord_to_box(ann['cord'])
            try:
                cls_id = int(self.cat_ids[ann['category_id']])
            except KeyError:
                print("Wrong label!!! ", file_name)
                continue

            if cord[0] <= cord[2]:
                x_left = cord[0]
                x_right = cord[2]
                y_left = cord[1]
                y_right = cord[3]
            else:    # cord[0] > cord[2]:
                x_left = cord[2]
                x_right = cord[0]
                y_left = cord[3]
                y_right = cord[1]
            cord[0] = x_left
            cord[1] = y_left
            cord[2] = x_right
            cord[3] = y_right

            if flipped:
                cord[[0, 2]] = width - cord[[2, 0]] - 1
                cord[[1, 3]] = cord[[3, 1]]

            cord[:2] = affine_transform(cord[:2], trans_output)
            cord[2:] = affine_transform(cord[2:], trans_output)

            direct_str = 'lt2rb' if cord[0] < cord[2] and cord[1] < cord[3] else 'lb2rt'

            if 0 < cord[0] < output_w and 0 < cord[2] < output_w \
                and 0 < cord[1] < output_h and 0 < cord[3] < output_h:
                if cord[0] == cord[2]:    # vertical line
                    angle = 90
                else:
                    a = (cord[1] - cord[3]) / (cord[0] - cord[2])
                    angle = np.arctan(a) * 180 / 3.14159265359
                pass
            else:
                if cord[0] == cord[2]:    # vertical line
                    if cord[0] < 0 or cord[0] >= output_w:
                        continue
                    cord[[1, 3]] = np.clip(cord[[1, 3]], 0, output_h - 1)
                    if cord[1] == cord[3]:
                        continue
                elif cord[1] == cord[3]:    # horizontal line
                    if cord[1] < 0 or cord[1] >= output_h:
                        continue
                    cord[[0, 2]] = np.clip(cord[[0, 2]], 0, output_w - 1)
                    if cord[0] == cord[2]:
                        continue
                else:
                    a = (cord[1] - cord[3]) / (cord[0] - cord[2])
                    b = (cord[0]*cord[3] - cord[2]*cord[1]) / (cord[0] - cord[2])

                    # Clip y first, then update x.
                    x0, y0, x1, y1 = cord[[0, 1, 2, 3]]
                    (y0, y1) = np.clip((y0, y1), 0, output_h - 1)
                    if y0 == y1:
                        continue
                    if y0 != cord[1]:
                        x0 = (y0 - b) / a
                    if y1 != cord[3]:
                        x1 = (y1 - b) / a
                    # Then clip x, then update y:
                    (x0, x1) = np.clip((x0, x1), 0, output_w - 1)
                    if x0 == x1:
                        continue
                    if x0 != cord[0]:
                        y0 = a * x0 + b
                    if x1 != cord[2]:
                        y1 = a * x1 + b

                    # Copy back to cord:
                    if direct_str == 'lt2rb':
                        cord[[0, 1, 2, 3]] = min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)
                    else:
                        cord[[0, 1, 2, 3]] = min(x0, x1), max(y0, y1), max(x0, x1), min(y0, y1)

            h, w = abs(cord[3] - cord[1]), abs(cord[2] - cord[0])
            w = 0.25 if w == 0 else w
            h = 0.25 if h == 0 else h
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))

                ct = np.array([(cord[0] + cord[2]) / 2, (cord[1] + cord[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                
                radius = max(0, int(radius))
                hm[cls_id] = draw_umich_gaussian(hm[cls_id], ct_int, radius)

                wh[k] = 1. * w, 1. * h
                direct[k] = 1 if direct_str == 'lt2rb' else 0
                direct2append = direct[k]

                ind[k] = ct_int[1] * output_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1
                gt_line.append([cord[0], cord[1], cord[2], cord[3], 1, cls_id, direct2append])
        
        ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind}
        ret.update({'wh': wh})
        ret.update({'direct': direct})
        
        if self.opt.reg_offset:
            ret.update({'reg': reg})

        if self.opt.loss_hm_magnitude:
            ret.update({'grad_magnitude': inp_grad_magnitude})
        if self.opt.debug > 0 or self.split == 'test':
            gt_line = np.array(gt_line, dtype=np.float32) if len(gt_line) > 0 else \
                             np.zeros((1, 7), dtype=np.float32)
            meta = {'c': c, 's': s, 'gt_line': gt_line, 'img_id': img_id}
            ret['meta'] = meta

        return ret
