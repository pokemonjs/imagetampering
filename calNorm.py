import json
import numpy as np
import copy
import cv2
import os
import sys
import torch

image_root = r"D:\Pokemonjs\Work\FAKEDATA\norm_img"  # 'db', 'coco', 'images')

from os import listdir
from os.path import isfile, join

import pathlib
# print(pathlib.Path('yourPath.example').suffix) # this will give result  '.example'

onlyfiles = [f for f in listdir(image_root) if isfile(join(image_root, f)) ]
my_imgfiles = []
for i  in range(len(onlyfiles)):
    if(pathlib.Path(onlyfiles[i]).suffix=='.png') or (pathlib.Path(onlyfiles[i]).suffix=='.jpg') or \
            (pathlib.Path(onlyfiles[i]).suffix == '.jpeg') or (pathlib.Path(onlyfiles[i]).suffix=='.tif'):
        my_imgfiles.append(onlyfiles[i])

pix = 1.0
sum_tot = torch.tensor([0, 0, 0])
sum_sq_tot = torch.tensor([0, 0, 0])
sum_pix = 0.0
for i in range(len(my_imgfiles)):
    img_name = os.path.join(image_root, my_imgfiles[i])

    # NOTE: must convert to "int", otherwise image**2 will overflow 
    #       since results are limited to uint8 [0~255]
    # 
    image_cv = cv2.imread(img_name) / 255.
    # image_cv = image_cv.astype(int)

    image = torch.from_numpy(image_cv)
    if (3 != image.shape[2]):
        continue
    sum = torch.sum(image, dim=[0, 1])
    sumsq = torch.sum(image ** 2, dim=[0, 1])
    pix = image.shape[0] * image.shape[1]

    # NOTE: you can test each image with the following codes (it should be: std1 == std2)
    # avg = sum/pix    
    # image_avg = torch.ones(image.shape) * avg
    # image_dif = image - image_avg
    # std1 = torch.sum(image_dif**2, [0, 1])/(pix - 1)
    # std2 = (sumsq - pix* avg**2)/(pix -1)

    sum_tot = sum_tot + sum
    sum_sq_tot = sum_sq_tot + sumsq
    sum_pix = sum_pix + pix
    if i % 1000 == 0:
        print(i, '/', len(my_imgfiles))

mean = sum_tot / sum_pix
std = (sum_sq_tot - sum_pix * mean ** 2) / (sum_pix - 1.0)
std = std ** 0.5

# print(mean,std)
print(list(mean.numpy()),list(std.numpy()))
# SAN: tensor([0.4123, 0.4481, 0.4420], dtype=torch.float64) tensor([0.2862, 0.2553, 0.2642], dtype=torch.float64)
# CASIA: [0.39225175 0.43358087 0.44139242] [0.28549757 0.26002468 0.26558968]
# Columbia:
# NIST16:
# IMD20:
# COVERAGE: [0.39014999 0.45414207 0.49376655] [0.25119353 0.25717758 0.2662189 ]