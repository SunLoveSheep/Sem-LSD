from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

from models.losses import FocalLoss, CrossEntropyLoss, FocalLossMagnitude, \
    FocalLossMagnitudePosOnly, FocalLossMagnitudeNegOnly
from models.losses import RegL1Loss, RegLoss
from models.decode import ctdet_line_decode
from models.utils import _sigmoid
from utils.debugger import Debugger
from utils.post_process import ctdetline_post_process
from .base_trainer import BaseTrainer


class CtdetLoss(torch.nn.Module):
  def __init__(self, opt):
    super(CtdetLoss, self).__init__()
    self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
    self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
              RegLoss() if opt.reg_loss == 'sl1' else None
    self.crit_wh = self.crit_reg

    if opt.direct_loss == 'cls':
      self.crit_direct = CrossEntropyLoss()
    else:
      self.crit_direct = RegL1Loss()
    self.opt = opt

    if opt.loss_hm_magnitude:
        if opt.loss_hm_magnitude_pos_only:
            self.crit_grad_magnitude = FocalLossMagnitudePosOnly()
        elif opt.loss_hm_magnitude_neg_only:
            self.crit_grad_magnitude = FocalLossMagnitudeNegOnly()
        else:
            self.crit_grad_magnitude = FocalLossMagnitude()

  def forward(self, outputs, batch):
    opt = self.opt
    hm_loss, off_loss = 0, 0
    if opt.loss_hm_magnitude:
        hm_grad_magnitude_loss = 0
    wh_loss = 0
    direct_loss = 0

    for s in range(opt.num_stacks):
      output = outputs[s]
      if isinstance(output, list):
          output = output[0]

      if not opt.mse_loss:
        output['hm'] = _sigmoid(output['hm'])

      hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks

      if opt.loss_hm_magnitude:
          output['grad_magnitude'] = _sigmoid(output['grad_magnitude'])
          hm_grad_magnitude_loss += self.crit_grad_magnitude(output['grad_magnitude'], batch['grad_magnitude']) \
                                    / opt.num_stacks

      if opt.reg_offset and opt.off_weight > 0:
        # print("\n loss off")
        off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                             batch['ind'], batch['reg']) / opt.num_stacks
          
      l1 = self.crit_wh(
        output['wh'], batch['reg_mask'],
        batch['ind'], batch['wh']) / opt.num_stacks
      wh_loss += l1

      direct_loss += self.crit_direct(
          output['direct'], batch['reg_mask'],
          batch['ind'], batch['direct']) / opt.num_stacks

    loss = opt.hm_weight * hm_loss + \
             opt.off_weight * off_loss + opt.direct_weight * direct_loss
    loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                  'off_loss': off_loss,
                  'direct_loss': direct_loss}
    if opt.loss_hm_magnitude:
        loss += opt.hm_magnitude_weight * hm_grad_magnitude_loss
        loss_stats.update({'grad_mag_loss': hm_grad_magnitude_loss})
    loss += opt.wh_weight * wh_loss
    loss_stats.update({'wh_loss': wh_loss})

    return loss, loss_stats


class CtdetLineTrainer(BaseTrainer):
  def __init__(self, opt, model, optimizer=None):
    super(CtdetLineTrainer, self).__init__(opt, model, optimizer=optimizer)

  def _get_losses(self, opt):
    loss_states = ['loss', 'hm_loss', 'off_loss', 'wh_loss', 'direct_loss']
    if opt.loss_hm_magnitude:
      loss_states.append('grad_mag_loss')
    loss = CtdetLoss(opt)
    return loss_states, loss

  def debug(self, batch, output, iter_id):
    opt = self.opt
    reg = output['reg'] if opt.reg_offset else None

    directs = output['direct']
    dets = ctdet_line_decode(
        output['hm'], output['wh'], reg=reg, directs=directs,
        cat_spec_wh=opt.cat_spec_wh, K=opt.K, direct_loss=opt.direct_loss)

    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets[:, :, :4] *= opt.down_ratio

    if not self.opt.temporal_model and not opt.no_reconstruct_loss:
      dets_gt = batch['meta']['gt_line'].numpy().reshape(1, -1, dets.shape[2] - 1 + 1)  # +1 for direction
      dets_gt[:, :, :4] *= opt.down_ratio
      for i in range(1):
        debugger = Debugger(
          dataset=opt.dataset, ipynb=(opt.debug==3), theme=opt.debugger_theme)
        img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
        img = np.clip(((
          img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
        pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
        gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
        debugger.add_blend_img(img, pred, 'pred_hm')
        debugger.add_blend_img(img, gt, 'gt_hm')
        debugger.add_img(img, img_id='out_pred')
        for k in range(len(dets[i])):
          if dets[i, k, 4] > opt.center_thresh:
            debugger.add_bbox_line(dets[i, k, :4], dets[i, k, -1],
                                   dets[i, k, 4], img_id='out_pred',
                                   direct=dets[i, k, 5])

        debugger.add_img(img, img_id='out_gt')
        for k in range(len(dets_gt[i])):
          if dets_gt[i, k, 4] > opt.center_thresh:
            debugger.add_bbox_line_gt(dets_gt[i, k, :4], dets_gt[i, k, 5],
                                      dets_gt[i, k, 4], img_id='out_gt',
                                      direct=dets_gt[i, k, -1])  # direct=directs_gt[i, k])

        if opt.debug == 4:
          debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
        else:
          debugger.show_all_imgs(pause=True)

  def save_result(self, output, batch, results):
    reg = output['reg'] if self.opt.reg_offset else None
    directs = output['direct']
    dets = ctdet_line_decode(
        output['hm'], output['wh'], reg=reg, directs=directs,
        cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets_out = ctdetline_post_process(
      dets.copy(), batch['meta']['c'].cpu().numpy(),
      batch['meta']['s'].cpu().numpy(),
      output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
    results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]
