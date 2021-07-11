from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import torch
# import torch.nn as nn
import paddle

# 来自于X2paddle
class Gather(object):
    def __init__(self, dim):
        self.dim = dim
        self.dtype_mapping = {"VarType.INT32": "int32", 
                              "paddle.int64" : "int64",
                              "paddle.int32" : "int32",
                              "VarType.INT64": "int64"}
        
    def __call__(self, x, index):
        if self.dim < 0:
            self.dim += len(x.shape)
        x_range = list(range(len(x.shape)))
        x_range[0] = self.dim
        x_range[self.dim] = 0
        x_swaped = paddle.transpose(x, perm=x_range)
        index_range = list(range(len(index.shape)))
        index_range[0] = self.dim
        index_range[self.dim] = 0
        index_swaped = paddle.transpose(index, perm=index_range)
        dtype = self.dtype_mapping[str(index.dtype)]
        
        x_shape = paddle.shape(x_swaped)
        index_shape = paddle.shape(index_swaped)
        
        prod = paddle.prod(x_shape, dtype=dtype) / x_shape[0]
        
        x_swaped_flattend = paddle.flatten(x_swaped)
        index_swaped_flattend = paddle.flatten(index_swaped)
        index_swaped_flattend *= prod
        
        bias = paddle.arange(start=0, end=prod, dtype=dtype)
        bias = paddle.reshape(bias, x_shape[1:])
        bias = paddle.crop(bias, index_shape[1:])
        bias = paddle.flatten(bias)
        bias = paddle.tile(bias, [index_shape[0]])
        index_swaped_flattend += bias
        
        gathered = paddle.index_select(x_swaped_flattend, index_swaped_flattend)
        gathered = paddle.reshape(gathered, index_swaped.shape)
        
        out = paddle.transpose(gathered, perm=x_range)

        return out

def _sigmoid(x):
  # y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  y = paddle.clip(paddle.nn.Sigmoid()(x), min=1e-4, max=1 - 1e-4)
  return y

def _gather_feat(feat, ind, mask=None):
    # dim  = feat.size(2)
    dim  = feat.shape[2]
    # ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    ind = ind.unsqueeze(2).expand(shape=[ind.shape[0], ind.shape[1], dim])
    # feat = feat.gather(1, ind)
    gather = Gather(1)
    feat = gather(feat, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        # feat = feat.view(-1, dim)
        feat = feat.reshape([-1, dim])
    return feat

def _tranpose_and_gather_feat(feat, ind):
    # feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.transpose(perm=[0, 2, 3, 1])
    # feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = feat.reshape([feat.shape[0], -1, feat.shape[3]])
    feat = _gather_feat(feat, ind)
    return feat

# 没用到过
# def flip_tensor(x):
#     return torch.flip(x, [3])
#     # tmp = x.detach().cpu().numpy()[..., ::-1].copy()
#     # return torch.from_numpy(tmp).to(x.device)

# def flip_lr(x, flip_idx):
#   tmp = x.detach().cpu().numpy()[..., ::-1].copy()
#   shape = tmp.shape
#   for e in flip_idx:
#     tmp[:, e[0], ...], tmp[:, e[1], ...] = \
#       tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
#   return torch.from_numpy(tmp.reshape(shape)).to(x.device)

# def flip_lr_off(x, flip_idx):
#   tmp = x.detach().cpu().numpy()[..., ::-1].copy()
#   shape = tmp.shape
#   tmp = tmp.reshape(tmp.shape[0], 17, 2, 
#                     tmp.shape[2], tmp.shape[3])
#   tmp[:, :, 0, :, :] *= -1
#   for e in flip_idx:
#     tmp[:, e[0], ...], tmp[:, e[1], ...] = \
#       tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
#   return torch.from_numpy(tmp.reshape(shape)).to(x.device)