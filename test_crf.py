import os
import cv2
import numpy as np
from collections import Counter
import paddle
from rrunet.config import *
from crf_infer import *
from rrunet.unet_model import *
from dataset import *
import sys
sys.path.append("PaddleSeg")
from paddleseg.transforms import Compose

import pydensecrf.densecrf as dcrf
dcrf.DenseCRF2D(256, 256, 2)

img=cv2.imread("out_erode.jpg")
img=cv2.imread("data/CASIA1/gt/Sp/Sp_S_NRN_R_art0056_art0056_0086_gt.png",0)
img=cv2.imread(r'C:\Users\Administrator\Desktop\HH\FAKEDATA\CASIA\val\mask\Sp_D_CND_A_pla0005_pla0023_0281_gt.png')
img=cv2.imread(r'C:\Users\Administrator\Desktop\HH\NewData1\train\mask\1cp_.png')
img=cv2.imread("data/CASIA2/gt/Tp_D_CND_M_N_ani00018_sec00096_00138_gt.png",0)
img=cv2.imread(r'D:\Pokemonjs\Work\data\test\Columbia\ca_se_att_rrunet\2.jpg',0)
# print(np.bincount(img.flatten()))
print(Counter(list(img.flatten())))
print(img.shape)

file_name = file_names[index if index!=14 else 7]#.replace("san","san_maxf1")
model = models[index]
save_dir=f"rrunet_pth/{file_name.split('.')[1]}"
dataset="CASIA"
mode="val"
path = f"C:/Users/Administrator/Desktop/HH/FAKEDATA/{dataset}/{mode}/images"
fix = "_gt" if "casia" in dataset.lower() else ""
path = [f"{path}/{p} {path.replace('images', 'mask')}/{p[:-4]}{fix}.png" for p in os.listdir(path)][0]


layer_state_dict = paddle.load("{}/{}".format(save_dir, file_name.replace("maxf1", "").replace(".pdparams",
                                                                                               ".finetune.pdparams").replace(
    "san", dataset)))
print("{}/{}".format(save_dir,
                     file_name.replace("maxf1", "").replace(".pdparams", ".finetune.pdparams").replace("san", dataset)))

model.set_state_dict(layer_state_dict)
model.eval()

image, label = path.split(" ")
# image=cv2.imread(image.strip())
if not os.path.exists(image) or not os.path.exists(label):
	print("not found:", image, label)
image = cv2.imread(image.strip()).astype("float32")
label = cv2.imread(label.strip(), 0)  # .replace(".png","_gt.png")
print(Counter(list(label.flatten())))
image = (image) / 255.
image = cv2.resize(image, (size, size))
label = cv2.resize(label, (size, size), interpolation=cv2.INTER_NEAREST)
print(Counter(list(label.flatten())))
cv2.imwrite("label.png",label)
# label=label.reshape((1,256,256))#.astype("int64")
label[label > 1] = 1
transform_ = Compose([])
x, y = (paddle.to_tensor(np.array([image.transpose((2,1,0))])), paddle.to_tensor(np.array([label])))
y = y.astype(np.float32)

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

pred = model(x)[0].numpy().reshape((256,256))
p = sigmoid(pred)
print(Counter(list(p.flatten())))
p[p>0.5]=255
p[p<=0.5]=0
cv2.imwrite("pred.png",p)
pred = dense_crf(image.astype(np.uint8),pred)
pred = sigmoid(pred)
print(Counter(list(pred.flatten())))
pred[pred>0.5]=255
pred[pred<=0.5]=0
cv2.imwrite("pred_crf.png",pred)