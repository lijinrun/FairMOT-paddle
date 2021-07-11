from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
# import torch
import numpy as np
# import torch.nn as nn
# import torch.nn.functional as F
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import KaimingUniform, Uniform
# from fvcore.nn import sigmoid_focal_loss_jit

from models.losses import FocalLoss#, TripletLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from models.decode import mot_decode
from models.utils import _sigmoid, _tranpose_and_gather_feat
from utils.post_process import ctdet_post_process
from .base_trainer import BaseTrainer


class MotLoss(nn.Layer):
    def __init__(self, opt):
        super(MotLoss, self).__init__()
        self.crit = paddle.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        self.crit_wh = paddle.nn.L1Loss(reduction='sum') if opt.dense_wh else \
            NormRegL1Loss() if opt.norm_wh else \
                RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
        self.opt = opt
        self.emb_dim = opt.reid_dim
        self.nID = opt.nID

        # param_attr = paddle.ParamAttr(initializer=KaimingUniform())
        # bound = 1 / math.sqrt(self.emb_dim)
        # bias_attr = paddle.ParamAttr(initializer=Uniform(-bound, bound))
        # self.classifier = nn.Linear(self.emb_dim, self.nID, weight_attr=param_attr, bias_attr=bias_attr)
        self.classifier = nn.Linear(self.emb_dim, self.nID, bias_attr=True)
        if opt.id_loss == 'focal': # 一般用不到
            # torch.nn.init.normal_(self.classifier.weight, std=0.01)
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            # torch.nn.init.constant_(self.classifier.bias, bias_value)

            weight_attr = paddle.framework.ParamAttr(initializer=nn.initializer.Normal(std=0.01))
            bias_attr = paddle.framework.ParamAttr(initializer=nn.initializer.Constant(bias_value))
            self.classifier = nn.Linear(self.emb_dim, self.nID, weight_attr=weight_attr, bias_attr=bias_attr)
        self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)
        self.emb_scale = math.sqrt(2) * math.log(self.nID - 1)
        # self.s_det = nn.Parameter(-1.85 * torch.ones(1))
        # self.s_id = nn.Parameter(-1.05 * torch.ones(1))
        self.s_det = paddle.create_parameter([1], dtype='float32', default_initializer = nn.initializer.Constant(value=-1.85))
        self.s_id = paddle.create_parameter([1], dtype='float32', default_initializer = nn.initializer.Constant(value=-1.05))
    def forward(self, outputs, batch):
        opt = self.opt
        hm_loss, wh_loss, off_loss, id_loss = 0, 0, 0, 0
        for s in range(opt.num_stacks):
            output = outputs[s]
            if not opt.mse_loss:
                output['hm'] = _sigmoid(output['hm'])

            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
            if opt.wh_weight > 0:
                wh_loss += self.crit_reg(
                    output['wh'], batch['reg_mask'],
                    batch['ind'], batch['wh']) / opt.num_stacks

            if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                          batch['ind'], batch['reg']) / opt.num_stacks

            if opt.id_weight > 0:
                id_head = _tranpose_and_gather_feat(output['id'], batch['ind'])
                # id_head = id_head[batch['reg_mask'] > 0].contiguous()
                id_head = paddle.to_tensor(id_head.numpy()[batch['reg_mask'].numpy()>0])
                # id_head = paddle.masked_select(id_head, batch['reg_mask'] > 0)                
                id_head = self.emb_scale * F.normalize(id_head)
                # id_target = batch['ids'][batch['reg_mask'] > 0]
                id_target = paddle.to_tensor(batch['ids'].numpy()[batch['reg_mask'].numpy() > 0])
                # id_target = paddle.masked_select(batch['ids'], batch['reg_mask'] > 0)
                id_output = self.classifier(id_head)#.contiguous()
                id_target.stop_gradient = True
                id_loss += self.IDLoss(id_output, id_target)

        det_loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss
        if opt.multi_loss == 'uncertainty':
            # loss = torch.exp(-self.s_det) * det_loss + torch.exp(-self.s_id) * id_loss + (self.s_det + self.s_id)
            loss = paddle.exp(-self.s_det) * det_loss + paddle.exp(-self.s_id) * id_loss + (self.s_det + self.s_id)
            loss *= 0.5
            
        else:
            loss = det_loss + 0.1 * id_loss

        loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                      'wh_loss': wh_loss, 'off_loss': off_loss, 'id_loss': id_loss}
        id_classifier_state_dict = self.classifier.state_dict()
        return loss, loss_stats, id_classifier_state_dict


class MotTrainer(BaseTrainer):
    def __init__(self, opt, model):
        super(MotTrainer, self).__init__(opt, model)

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss', 'id_loss']
        loss = MotLoss(opt)
        return loss_states, loss

    def save_result(self, output, batch, results):
        reg = output['reg'] if self.opt.reg_offset else None
        dets = mot_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets_out = ctdet_post_process(
            dets.copy(), batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().numpy(),
            output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]
