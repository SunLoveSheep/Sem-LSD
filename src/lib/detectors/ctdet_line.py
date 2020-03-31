from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import torch

from external_nms.nms import soft_nms
from models.decode import ctdet_line_decode
from models.utils import flip_tensor
from utils.post_process import ctdetline_post_process

from .base_detector import BaseDetector


class CtdetLineDetector(BaseDetector):
  def __init__(self, opt):
    super(CtdetLineDetector, self).__init__(opt)
  
  def process(self, images, return_time=False):
    with torch.no_grad():
      output = self.model(images)[-1]
      hm = output['hm'].sigmoid_()
      wh = output['wh']
      directs = output['direct']
      reg = output['reg'] if self.opt.reg_offset else None
      if self.opt.flip_test:
        hm = (hm[0:1] + flip_tensor(hm[1:2])) / 2
        wh = (wh[0:1] + flip_tensor(wh[1:2])) / 2
        reg = reg[0:1] if reg is not None else None
      if 'cpu' not in self.opt.device.type:
        torch.cuda.synchronize()
      forward_time = time.time()
      dets = ctdet_line_decode(hm, wh, reg=reg, directs=directs, K=self.opt.K, direct_loss=self.opt.direct_loss)
      
    if return_time:
      return output, dets, forward_time
    else:
      return output, dets

  def post_process(self, dets, meta, scale=1):
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])

    dets = ctdetline_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], self.opt.num_classes)
    for j in range(1, self.num_classes + 1):
      dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 6)
      dets[0][j][:, :4] /= scale
    return dets[0]

  def merge_outputs(self, detections):
    results = {}
    for j in range(1, self.num_classes + 1):
      results[j] = np.concatenate(
        [detection[j] for detection in detections], axis=0).astype(np.float32)
      if len(self.scales) > 1 or self.opt.nms:
         soft_nms(results[j], Nt=0.5, method=2)
    scores = np.hstack(
      [results[j][:, 4] for j in range(1, self.num_classes + 1)])
    if len(scores) > self.max_per_image:
      kth = len(scores) - self.max_per_image
      thresh = np.partition(scores, kth)[kth]
      for j in range(1, self.num_classes + 1):
        keep_inds = (results[j][:, 4] >= thresh)
        results[j] = results[j][keep_inds]
    return results

  def debug(self, debugger, images, dets, output, scale=1):
    detection = dets.detach().cpu().numpy().copy()
    detection[:, :, :4] *= self.opt.down_ratio
    for i in range(1):
      img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
      img = ((img * self.std + self.mean) * 255).astype(np.uint8)
      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(scale))
      debugger.add_img(img, img_id='out_pred_{:.1f}'.format(scale))
      for k in range(len(dets[i])):
        if detection[i, k, 4] > self.opt.center_thresh:
          debugger.add_bbox_line(detection[i, k, :4], detection[i, k, -1],
                                 detection[i, k, 4],
                                 img_id='out_pred_{:.1f}'.format(scale),
                                 direct=detection[i, k, 5])

  def show_results(self, debugger, image, results, cnt, img_name):
    debugger.add_img(image, img_id='ctdet_line')
    for j in range(1, self.num_classes + 1):
      vis_thresh = self.opt.vis_thresh
      if j == 2 and self.opt.task == 'ctdet_line':  # For ground_building type
        vis_thresh = self.opt.vis_thresh_gbuilding if self.opt.vis_thresh_gbuilding > 0 else self.opt.vis_thresh
      for bbox in results[j]:
        if bbox[4] > vis_thresh:
          debugger.add_bbox_line(bbox[:4], j - 1, bbox[4], img_id='ctdet_line', direct=bbox[5])
    # debugger.show_all_imgs(pause=self.pause)
    if self.opt.save_img == 1:
      debugger.save_all_imgs(path=self.opt.save_path, prefix='{}'.format(self.opt.arch),
                             cnt=cnt, img_name=img_name)
