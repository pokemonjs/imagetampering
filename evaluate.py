import os, sys

sys.path.append("PaddleSeg")
sys.path.append("../")
print(os.getcwd())
from PaddleSeg import paddleseg
import random
import paddle
import time
import numpy as np
import paddle.nn as nn
from paddle.io import DataLoader, Dataset
import paddle.nn.functional as F
import glob
from sklearn.metrics import roc_auc_score
from paddle.regularizer import L1Decay,L2Decay
from dataset import *
from rrunet.cosinewarmup import *
from utils import *
from metrics import *

paddle.seed(123)
np.random.seed(2021)
random.seed(123)

from rrunet.config import *

iouw = 1
train_eval=True
train_eval=False
if len(sys.argv)>1:
    index = int(sys.argv[1])
if len(sys.argv)>2:
    dataset = sys.argv[2]
if len(sys.argv)>3:
    iouw = float(sys.argv[3])
if len(sys.argv)>4:
    train_eval = int(sys.argv[4])==1
print("train_eval",train_eval)
file_name = file_names[index]#.replace("san","sanmaxf1")
print("eval:",file_name)
# if index!=14 else 7
model = models[index]

save_dir=f"rrunet_pth/{file_name.split('.')[1]}"

# flops = paddle.flops(model, [1,3,128,128], print_detail=True)

# print(file_name,model)
ratio = 0.9
# os.system("rm -rf test/casia_eval/pre/*")
# os.system("rm -rf test/casia_eval/gt/*")
paddle.utils.run_check()
lr_base = 1e-3
training = False
print(file_name)
loss_fn = paddle.nn.CrossEntropyLoss(axis=1)  #
loss_fn = paddle.nn.BCELoss()
loss_fn = BCEIOULoss(iouw)
# loss_fn = BCEDiceLoss()
#修改
epochs=30
lr = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=lr_base, T_max=epochs)
# lr = CosineWarmup(lr=lr_base,step_each_epoch=100,epochs=epochs,warmup_epoch=5)
if not training:
    epochs = 0
# opt=paddle.optimizer.Momentum(learning_rate=lr,parameters=model.parameters(),weight_decay=1e-2)
opt = paddle.optimizer.Adam(learning_rate=lr, parameters=model.parameters(), weight_decay=L2Decay(1e-4))  # ,weight_decay=1e-5



# finetune_train_dataloader = DataLoader(NewDataset("train", dataset, ratio), batch_size=batch_size, shuffle=True)
# 测试用
# finetune_eval_dataloader = DataLoader(NewDataset("test", dataset, ratio), batch_size=1, shuffle=False)
# finetune_eval_dataloader = DataLoader(NewDataset("val", dataset, ratio), batch_size=1, shuffle=False)

finetune_train_dataloader = DataLoader(FinetuneDataset("train", dataset, ratio), batch_size=batch_size, shuffle=True)
finetune_eval_dataloader = DataLoader(FinetuneDataset("train", dataset, ratio), batch_size=1, shuffle=False)

# #修改
# finetune_eval_dataloader = DataLoader(TestDataset("train" if "casia1" in eval_set else "val", eval_set, ratio), batch_size=1, shuffle=False)

# print("dataset")

# list_set = open("data/list_{}.txt".format(dataset)).readlines()
# list_set = list_set[int(len(list_set) * ratio):]
# val_set = [v.strip().split(" ")[-1] for v in list_set]
# val_img_set = [v.strip().split(" ")[0] for v in list_set]

with open("{}/f1.txt".format(save_dir.split("/")[0]), "a") as f:
    f.write("eval:{}\n".format(file_name))
    f.write("dataset:{} eval_set:{}\n".format(dataset,eval_set))


# 验证集测试
# 模型推理 基础API
# load
# print(save_dir)
layer_state_dict = paddle.load("{}/{}".format(save_dir, file_name.replace("maxf1", "").replace(".pdparams",
                                                                                               ".finetune.pdparams").replace(
    "san", dataset)))
print("{}/{}".format(save_dir,
                     file_name.replace("maxf1", "").replace(".pdparams", ".finetune.pdparams").replace("san", dataset)))

# layer_state_dict = paddle.load("{}/{}".format(save_dir, "final.resunet.columbia_aug.finetune - 副本.pdparams"))
layer_state_dict = paddle.load("rrunet_pth/dual_ca_se_afusion_rrunet/final.dual_ca_se_afusion_rrunet.CASIA.finetune_real561.pdparams")
model.set_state_dict(layer_state_dict)
# model.eval() if index!=7 else model.train()
model.eval()
print("model eval")

# st = "CASIA1" if dataset == "casia1_splice" else "Columbia"
path = "data/test/{}/{}".format(eval_set, file_name.split(".")[1])
if not os.path.exists(path):
    os.mkdir(path)

ind = 0
TP, FP, FN, TN = 0, 0, 0, 0
# adam.set_state_dict(opt_state_dict)
AUC = 0
# model.eval()
pre = 0
rec = 0
f1_avg = 0
auc_pred = []
auc_label = []
for batch_id, data in enumerate(finetune_eval_dataloader):
    TP, FP, FN, TN = 0, 0, 0, 0

    x_data = data[0]  # 测试数据
    # x_data = paddle.unsqueeze(x_data,axis=0)
    label = data[1][0].numpy()  # 测试数据标签
    # label h,w
    path = data[2][0]
    predicts = model(x_data)[0].numpy()  # 预测结果
    # pred 1,h,w
    try:
        auc = roc_auc_score(label.flatten(), predicts.flatten())
        if auc>0.95 or True:
            auc_pred.append(predicts.flatten())
            auc_label.append(label.flatten())
    except ValueError as e:
        print(e)
        print("auc error:", path.split(" ")[-1])
        auc = 1
    # auc = 1
    # print("auc:", auc)
    # print(predicts.shape)
    # 计算损失与精度
    # loss = loss_fn(predicts, y_data)
    predicts = sigmoid(predicts)

    # pre = np.copy(predicts)
    # pre[pre>0.5]=1
    # pre[pre<=0.5]=0
    # cv2.imwrite("out.jpg", 255 * pre.transpose((2, 1, 0)))
    # predicts = erode(predicts)

    # predicts = paddle.nn.functional.sigmoid(predicts)
    predicts[predicts > 0.5] = 1
    predicts[predicts <= 0.5] = 0

    # acc = paddle.metric.accuracy(predicts, y_data)
    pred = predicts#[0]
    # print(np.sum(np.float32(pred[0]*255)))

    TP += np.sum(np.array(pred * label))

    FP += np.sum(np.array(pred * (1 - label)))

    FN += np.sum(np.array((1 - pred) * label))
    # FN偏高，负的推理成正的了

    TN += np.sum(np.array((1 - pred) * (1 - label)))
    
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    
    if auc> 0.9 and precision>0.9:
        print(path.split(" ")[0].split("/")[-1], auc, TP / (TP + FP), TP / (TP + FN))
    # pre += TP / (TP + FP)
    # rec += TP / (TP + FN)
    # f1_avg += 2 *(TP / (TP + FP) * TP / (TP + FN)) / ((TP / (TP + FP)) + (TP / (TP + FN)))
    # print(TP,FP,FN,TN)
    img_name = path.split(" ")[0].split("/")[-1]
    vis_path = "data/test/{}/{}/{}.png".format(eval_set, file_name.split(".")[1], img_name[:-4])
    # cv2.imwrite(vis_path, np.float32(pred[0] * 255))
    cv2.imwrite(vis_path, cv2.resize(np.float32(pred[0] * 255), (384, 256)))


    ind += 1
    # break

# FP += TP//7
# FN += TP//7
print("{},{},{},{}".format(TP, FP, FN, TN))
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * precision * recall / (precision + recall)
# print("val set size:", len(val_set))
auc_label=np.array(auc_label).flatten()
auc_pred=np.array(auc_pred).flatten()
auc_total = roc_auc_score(auc_label,auc_pred)
print("auc_total",auc_total)
print("precision:{}\nrecall:{}\nf1:{}\n".format(precision, recall, f1))
print(f"{pre/ind},{rec/ind},{f1_avg/ind}")
with open("{}/f1.txt".format(save_dir.split("/")[0]), "a") as f:
    f.write("auc_total:{}\n".format(auc_total))
    f.write("precision:{}\nrecall:{}\nf1:{}\n".format(precision, recall, f1))

# os.system("cd test/casia_eval && zip -q -r pre.zip pre")


auc_pred = []
auc_label = []
if train_eval:
    finetune_train_eval_dataloader = DataLoader(FinetuneDataset("val_train", dataset, ratio), batch_size=1,
                                                shuffle=False)
    next(finetune_train_dataloader())
    next(finetune_eval_dataloader())
    #  训练集验证
    ind = 0
    TP, FP, FN, TN = 0, 0, 0, 0
    # adam.set_state_dict(opt_state_dict)
    # model.eval()
    for batch_id, data in enumerate(finetune_train_eval_dataloader):

        x_data = data[0]  # 测试数据
        # x_data = paddle.unsqueeze(x_data,axis=0)
        label = data[1][0].numpy()  # 测试数据标签
        # label h,w
        path = data[2][0]
        predicts = model(x_data)[0].numpy()  # 预测结果
        # pred 1,h,w
        try:
            auc = roc_auc_score(label.flatten(), predicts.flatten())
            auc_pred.append(predicts.flatten())
            auc_label.append(label.flatten())
        except ValueError as e:
            print(e)
            print("auc error:", path.split(" ")[-1])
            auc = 1
        # auc = 1
        # print("auc:", auc)
        # print(predicts.shape)
        # 计算损失与精度
        # loss = loss_fn(predicts, y_data)
        predicts = sigmoid(predicts)

        # pre = np.copy(predicts)
        # pre[pre>0.5]=1
        # pre[pre<=0.5]=0
        # cv2.imwrite("out.jpg", 255 * pre.transpose((2, 1, 0)))
        # predicts = erode(predicts)

        # predicts = paddle.nn.functional.sigmoid(predicts)
        predicts[predicts > 0.5] = 1
        predicts[predicts <= 0.5] = 0

        # acc = paddle.metric.accuracy(predicts, y_data)
        pred = predicts#[0]
        # print(np.sum(np.float32(pred[0]*255)))

        TP += np.sum(np.array(pred * label))

        FP += np.sum(np.array(pred * (1 - label)))

        FN += np.sum(np.array((1 - pred) * label))
        # FN偏高，负的推理成正的了

        TN += np.sum(np.array((1 - pred) * (1 - label)))

        print(path.split(" ")[0].split("/")[-1], auc, TP / (TP + FP), TP / (TP + FN))
        # print(TP,FP,FN,TN)

        # path = "data/test/{}/{}/{}.jpg".format(eval_set, file_name.split(".")[1], data[2][0].split(" ")[0][:-4])
        # cv2.imwrite(path, cv2.resize(np.float32(pred[0] * 255), (384, 256)))

        ind += 1
        # break
    print("{},{},{},{}".format(TP, FP, FN, TN))
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    # print("val set size:", len(val_set))
    auc_label = np.array(auc_label).flatten()
    auc_pred = np.array(auc_pred).flatten()
    auc_total = roc_auc_score(auc_label, auc_pred)
    print("\ntrain_eval auc_total", auc_total)
    print("train_eval precision:{}\nrecall:{}\nf1:{}\n".format(precision, recall, f1))
    with open("{}/f1.txt".format(save_dir.split("/")[0]), "a") as f:
        f.write("train_eval auc_total:{}\n".format(auc_total))
        f.write("train_eval precision:{}\nrecall:{}\nf1:{}\n\n".format(precision, recall, f1))
