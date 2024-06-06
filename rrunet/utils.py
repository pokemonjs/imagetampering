import inspect
import os.path

import cv2
import paddle
import paddle.nn as nn
import numpy as np

import paddle.nn.functional as F
import re

from PIL import Image
from matplotlib import pyplot as plt

from rrunet import config


# import rrunet.config as config


class Sobel(nn.Layer):
    def __init__(self, stride, channels=3):
        super(Sobel, self).__init__()

        edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype="np.float")
        edge_kx = paddle.to_tensor(edge_kx)
        self.filt = paddle.repeat_interleave(edge_kx.reshape(1, 1, 3, 3), channels, 0)
        self.stride = stride

    def forward(self, x):
        out_x = F.conv2d(x, self.filt, stride=self.stride, padding=1, groups=x.shape[1])
        out_y = F.conv2d(x, self.filt.transpose((0, 1, 3, 2)), stride=self.stride, padding=1, groups=x.shape[1])
        out = paddle.sqrt(out_x ** 2 + out_y ** 2)

        return out


class SRM(nn.Layer):
    def __init__(self):
        super(SRM, self).__init__()
        q = [4.0, 12.0, 2.0]
        filter1 = [[0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 2, -4, 2, 0],
                   [0, -1, 2, -1, 0],
                   [0, 0, 0, 0, 0]]
        filter2 = [[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]]
        filter3 = [[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 1, -2, 1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]
        filter1 = np.asarray(filter1, dtype=float) / q[0]
        filter2 = np.asarray(filter2, dtype=float) / q[1]
        filter3 = np.asarray(filter3, dtype=float) / q[2]
        filters = [[filter1, filter1, filter1], [filter2, filter2, filter2], [filter3, filter3, filter3]]
        filters = np.asarray(filters, dtype=np.float32)

        SRM_kernel = paddle.to_tensor(filters)
        self.filter = SRM_kernel

    def forward(self, x):
        noise_fetures = F.conv2d(x, self.filter, stride=1, padding=2)

        return noise_fetures


def truncate_2(x):
    neg = ((x + 2) + paddle.abs(x + 2)) / 2 - 2
    return -(-neg + 2 + paddle.abs(- neg + 2)) / 2 + 2


def vis_feature(data, save_dir="vis_feature"):
    save_dir = "vis_feature/" + save_dir
    dirs = save_dir.split("/")
    for i in range(len(dirs)):
        pt = "/".join(dirs[:i + 1])
        # print(pt)
        if not os.path.exists(pt):
            os.mkdir(pt)
    if not os.path.exists(f"{dirs[0]}/{dirs[1]}/{dirs[2]}/res/"):
        os.mkdir(f"{dirs[0]}/{dirs[1]}/{dirs[2]}/res/")
    # data:c,h,w
    c, h, w = data.shape
    res = np.zeros((h, w))
    for i in range(c):
        if os.path.exists(f"{save_dir}/{i}.png"):
            # continue
            pass
        d = data[i]
        min_ = np.min(d)
        max_ = np.max(d)
        diff = max_ - min_
        if "weight" in save_dir:
            # print(f"{save_dir} {min_} {max_} {diff}")
            pass
        if diff != 0.0:
            d = (d - min_) / diff * 255.
        else:
            d = d * 255.
        d = cv2.resize(d, (256, 256))
        if "weight" not in save_dir:  # and "srm" not in save_dir
            d = cv2.applyColorMap(d.astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(f"{save_dir}/{i}.png", d)
    res = np.average(data, axis=0)
    min_ = np.min(res)
    max_ = np.max(res)
    diff = max_ - min_
    res = (res - min_) / diff * 255.
    res = cv2.resize(res, (256, 256))
    if "weight" not in save_dir and "srm" not in save_dir:
        res = cv2.applyColorMap(res.astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(f"{dirs[0]}/{dirs[1]}/{dirs[2]}/res/{dirs[-1]}.png", res)


def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)',line)
        if m:
            return m.group(1)

def vis_feature_plus(x, name):
    vis_feature(x.numpy()[0], f"{config.file_names[config.index].split('.')[1]}/{config.dataset}/{name}")
    # pass

def vis_feature_encoder(x1,x2,x3,x4,x5):
    vis_feature_plus(x1, varname(x1))
    vis_feature_plus(x2, varname(x2))
    vis_feature_plus(x3, varname(x3))
    vis_feature_plus(x4, varname(x4))
    vis_feature_plus(x5, varname(x5))

