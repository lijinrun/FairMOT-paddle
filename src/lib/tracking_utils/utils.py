import os.path as osp
import numpy as np
import os

#import maskrcnn_benchmark.layers.nms as nms
# Set printoptions
# torch.set_printoptions(linewidth=1320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5

def mkdir_if_missing(d):
    if not osp.exists(d):
        os.makedirs(d)


def float3(x):  # format floats to 3 decimals
    return float(format(x, '.3f'))

# 删掉了很多函数，都没用到过