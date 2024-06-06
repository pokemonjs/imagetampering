
import os,sys
sys.path.append("PaddleSeg")
print(os.getcwd())
# from PaddleSeg import paddleseg
import random
import paddle
import time
import numpy as np
import paddle.nn as nn
from paddle.io import DataLoader,Dataset
import paddle.nn.functional as F
import glob
from sklearn.metrics import roc_auc_score
from dataset import *
from config_models import models


paddle.seed(123)
np.random.seed(2021)
random.seed(123)

save_dir="rrunet_pth"



#切换model
file_names=[
    "final.unet.san.pdparams",#0
    "final.resunet.san.pdparams",#1
    "final.att_runet.san.pdparams",#2
    "final.se_att_runet.san.pdparams",#3
    "final.ca_se_att_runet.san.pdparams",#4
    "final.rrunet.san.pdparams",#5
    # "final.se_rrunet.san.pdparams",
    "final.se_att_rrunet_new.san.pdparams",#6
    "final.ca_se_att_rrunet.san.pdparams",#7
    "final.dse_att_rrunet.san.pdparams",#8
    "final.ca_att_runet.san.pdparams",#9
    "final.ca_att_rrunet.san.pdparams",#10
    "final.se_unet.san.pdparams",
    "final.ca_unet.san.pdparams",
    "final.ca_se_unet.san.pdparams",
    "final.ca_se_sa_rrunet.san.pdparams",
    "final.att_unet.san.pdparams",#15
    "final.se_rrunet.san.pdparams",
    "final.ca_se_fusion_rrunet.san.pdparams",
    "final.dual_ca_se_att_rrunet.san.pdparams",
    "final.ca_se_sa_rrunet.down1.san.pdparams",
    "final.ca_se_sa_rrunet.down2.san.pdparams", # 20
    "final.double_ca_se_att_rrunet.san.pdparams",
    "final.fusion_rrunet.san.pdparams",
    "final.ca_se_fusion_rrunet.san.pdparams",
    "final.dual_ca_se_fusion_rrunet.san.pdparams", # 24
    "final.dual_ca_se_afusion_rrunet.san.pdparams", # 25
    "final.dual_ca_se_dafusion_rrunet.san.pdparams", # 26
    "final.dual_ca_se_rafusion_rrunet.san.pdparams", # 27
    "final.dual_ca_se_afusion_rrunet2.san.pdparams", # 28
    "final.dual_ca_se_afusion_rrunet4.san.pdparams", # 29
    "final.dual_ca_se_afusionR_rrunet.san.pdparams", # 30
]
models=models


#切换模型
index = 25  # dual
# index = 7  # ca_se_att
# index = 5  # rrunet
batch_size = 16 #// 2
if index in [14, 19, 20]:
    batch_size = 16 // 4

#切换验证集
dataset = "casia1_splice"
# dataset = "columbia_aug"
dataset = "casia2"
dataset = "casia2_aug"
# dataset = "casia1_aug"
dataset = "CASIA"


# dataset = "NIST16"
# dataset = "NIST160"
# dataset = "NIST16_Clear"
#
#
# dataset = "JS_IMD2020"
# dataset = "JS_IMD2020_LARGE"
# dataset = "JS_COLUMBIA"
# dataset = "JS_COLUMBIA_ORI"
# dataset = "JS_COVERAGE"
# dataset = "JS_COVERAGE_AUG"
# dataset = "COVERAGE_AUG"

# dataset = "CASIA"
# eval_set = "casia1" if "casia2" in dataset else dataset
eval_set = "casia1_splice"
eval_set = dataset