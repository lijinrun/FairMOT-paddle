from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import torch
# import torch.nn as nn
import paddle
import paddle.nn as nn
from .utils import _gather_feat, _tranpose_and_gather_feat


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    # keep = (hmax == heat).float()
    keep = (hmax == heat).cast('float32')
    return heat * keep

# 没用到过
# def _topk_channel(scores, K=40):
#       # batch, cat, height, width = scores.size()

#       topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

#       topk_inds = topk_inds % (height * width)
#       topk_ys   = (topk_inds / width).int().float()
#       topk_xs   = (topk_inds % width).int().float()

#       return topk_scores, topk_inds, topk_ys, topk_xs

def _topk(scores, K=40):
    # batch, cat, height, width = scores.size()
    batch, cat, height, width = scores.shape
    # topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)
    topk_scores, topk_inds = paddle.topk(scores.reshape([batch, cat, -1]), K)
    topk_inds = topk_inds % (height * width)
    # topk_ys   = (topk_inds / width).int().float()
    topk_ys = (topk_inds / width).cast('int32').cast('float32')
    # topk_xs   = (topk_inds % width).int().float()
    topk_xs = (topk_inds % width).cast('int32').cast('float32')
    # topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_score, topk_ind = paddle.topk(topk_scores.reshape([batch, -1]), K)
    # topk_clses = (topk_ind / K).int()
    topk_clses = (topk_ind/K).cast('int32')
    # topk_inds = _gather_feat(
    #     topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_inds = _gather_feat(
        topk_inds.reshape([batch, -1, 1]), topk_ind).reshape([batch, K])
    # topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.reshape([batch, -1, 1]), topk_ind).reshape([batch, K])
    # topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.reshape([batch, -1, 1]), topk_ind).reshape([batch, K])
    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def mot_decode(heat, wh, reg=None, ltrb=False, K=100):
    # batch, cat, height, width = heat.size()
    batch, cat, height, width = heat.shape
    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)

    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
        reg = _tranpose_and_gather_feat(reg, inds)
        # reg = reg.view(batch, K, 2)
        # xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        # ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
        reg = reg.reshape([batch, K, 2])
        xs = xs.reshape([batch, K, 1]) + reg[:, :, 0:1]
        ys = ys.reshape([batch, K, 1]) + reg[:, :, 1:2]
    else:
        # xs = xs.view(batch, K, 1) + 0.5
        # ys = ys.view(batch, K, 1) + 0.5
        xs = xs.reshape([batch, K, 1]) + 0.5
        ys = ys.reshape([batch, K, 1]) + 0.5
    wh = _tranpose_and_gather_feat(wh, inds)
    if ltrb:
        # wh = wh.view(batch, K, 4)
        wh = wh.reshape([batch, K, 4])
    else:
        # wh = wh.view(batch, K, 2)
        wh = wh.reshape([batch, K, 2])
    # clses = clses.view(batch, K, 1).float()
    clses = clses.reshape([batch, K, 1]).cast('float32')
    # scores = scores.view(batch, K, 1)
    scores = scores.reshape([batch, K, 1])
    if ltrb:
        # bboxes = torch.cat([xs - wh[..., 0:1],
        #                     ys - wh[..., 1:2],
        #                     xs + wh[..., 2:3],
        #                     ys + wh[..., 3:4]], dim=2)
        bboxes = paddle.concat([xs - wh[:, :, 0:1],
                                ys - wh[:, :, 1:2],
                                xs + wh[:, :, 2:3],
                                ys + wh[:, :, 3:4]], axis=2).cast('float32')
    else:
        # bboxes = torch.cat([xs - wh[..., 0:1] / 2,
        #                     ys - wh[..., 1:2] / 2,
        #                     xs + wh[..., 0:1] / 2,
        #                     ys + wh[..., 1:2] / 2], dim=2)
        bboxes = paddle.concat([xs - wh[:, :, 0:1] / 2,
                                ys - wh[:, :, 1:2] / 2,
                                xs + wh[:, :, 0:1] / 2,
                                ys + wh[:, :, 1:2] / 2], axis=2).cast('float32')
    # detections = torch.cat([bboxes, scores, clses], dim=2)
    detections = paddle.concat([bboxes, scores, clses], axis=2)
    
    return detections, inds
