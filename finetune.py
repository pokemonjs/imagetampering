import os, sys

sys.path.append("PaddleSeg")
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
noatt=True
noatt=False
save_fn = ""
if len(sys.argv)>1:
    index = int(sys.argv[1])
if len(sys.argv)>2:
    dataset = sys.argv[2]
if len(sys.argv)>3:
    iouw = float(sys.argv[3])
if len(sys.argv)>4:
    noatt = int(sys.argv[4])==1
if len(sys.argv)>5:
    save_fn = sys.argv[5]

file_name = file_names[index]#.replace("san","sanmaxf1")
# if index!=14 else 7
model = models[index]

save_dir=f"rrunet_pth/{file_name.split('.')[1]}"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# flops = paddle.flops(model, [1,3,128,128], print_detail=True)

# print(file_name,model)
ratio = 0.9
# os.system("rm -rf test/casia_eval/pre/*")
# os.system("rm -rf test/casia_eval/gt/*")
paddle.utils.run_check()
lr_base = 1e-3
training = True
# training = False
resume = True  # if dataset=="casia1_splice" else False
# resume = False
epochs = 30
stop = True
stop_count = 0
stop_countT = 10
if resume:
    print("resume")
    print("noatt", noatt)
    if index == 25:
        layer_state_dict = paddle.load("{}/{}".format(save_dir, file_name))
        # layer_state_dict = paddle.load("{}/{}".format(save_dir, "final.dual_ca_se_fusion_rrunet.san.pdparams"))
        # 用预训练的afusion，但是no_att
        if noatt:
            # layer_state_dict = paddle.load("{}/{}".format(save_dir, file_name))
            remove = []
            for key in layer_state_dict.keys():
                if ".att." in key:
                    remove.append(key)
            for key in remove:
                del layer_state_dict[key]
    else:
        layer_state_dict = paddle.load("{}/{}".format(save_dir, file_name))
        if index == 27 and noatt:
            remove=[]
            for key in layer_state_dict.keys():
                if ".att." in key:
                    remove.append(key)
            for key in remove:
                del layer_state_dict[key]
            # pass
    # layer_state_dict = paddle.load("{}/{}".format(save_dir, "final.dual_ca_se_dafusion_rrunet.sanmaxf1.pdparams"))
    model.set_state_dict(layer_state_dict)
# if not training and not os.path.exists("{}/{}".format(save_dir,file_name.replace(".pdparams",".finetune.pdparams"))):
#     print("load")
#     paddle.save(model.state_dict(), "{}/{}".format(save_dir,file_name.replace(".pdparams",".finetune.pdparams")))
checkpoint = 8
print(file_name)
print(dataset)
loss_fn = paddle.nn.CrossEntropyLoss(axis=1)  #
loss_fn = paddle.nn.BCELoss()
loss_fn = BCEIOULoss(iouw)
# loss_fn = DBCEIOULoss(iouw)

# loss_fn = BCEDiceLoss()
#修改
lr = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=lr_base, T_max=epochs)
# lr = CosineWarmup(lr=lr_base,step_each_epoch=100,epochs=epochs,warmup_epoch=5)
if not training:
    epochs = 0
# opt=paddle.optimizer.Momentum(learning_rate=lr,parameters=model.parameters(),weight_decay=1e-2)
opt = paddle.optimizer.Adam(learning_rate=lr, parameters=model.parameters(), weight_decay=L2Decay(1e-4))  # ,weight_decay=1e-5
# model.set_state_dict(paddle.load("{}/final.pdparams".format(save_dir)))
# opt.set_state_dict(paddle.load("{}/final.pdopt".format(save_dir)))

# model.set_state_dict(paddle.load("rrunet_pth/dual_ca_se_afusion_rrunet/final.dual_ca_se_afusion_rrunet.69.pdparams"))
# finetune_train_dataloader = DataLoader(NewDataset("train", dataset, ratio), batch_size=batch_size, shuffle=True)
# 测试用
# finetune_eval_dataloader = DataLoader(NewDataset("test", dataset, ratio), batch_size=1, shuffle=False)
# finetune_eval_dataloader = DataLoader(NewDataset("val", dataset, ratio), batch_size=1, shuffle=False)
# #修改
# finetune_eval_dataloader = DataLoader(TestDataset("train" if "casia1" in eval_set else "val", eval_set, ratio), batch_size=1, shuffle=False)

finetune_train_dataloader = DataLoader(FinetuneDataset("train", dataset, ratio), batch_size=batch_size, shuffle=True)
finetune_eval_dataloader = DataLoader(FinetuneDataset("val", dataset, ratio), batch_size=1, shuffle=False)
# next(finetune_train_dataloader())
# next(finetune_eval_dataloader())
# print("dataset")

# list_set = open("data/list_{}.txt".format(dataset)).readlines()
# list_set = list_set[int(len(list_set) * ratio):]
# val_set = [v.strip().split(" ")[-1] for v in list_set]
# val_img_set = [v.strip().split(" ")[0] for v in list_set]

with open("{}/f1.txt".format(save_dir.split("/")[0]), "a") as f:
    f.write("finetune:{}\n".format(file_name))
    f.write("dataset:{} eval_set:{}\n".format(dataset,eval_set))

with open("{}/loss.txt".format(save_dir.split("/")[0]), "a") as f:
    f.write("finetune:{}\n".format(file_name))
    f.write("dataset:{} eval_set:{}\n".format(dataset,eval_set))

max_f1 = 0
max_auc = 0
for epoch in range(epochs):  # -checkpoint-1
    print(time.strftime("%d day %H:%M", time.localtime(time.time())))
    print("start epoch:{}/{}".format(epoch + 1, epochs))
    model.train()
    losses = [0]
    for ind, data in enumerate(finetune_train_dataloader):
        pass
        # 输入计算输出
        x, y = data[:2]
        yhat = model(x)
        yhat = paddle.nn.functional.sigmoid(yhat)
        # shape([x,yhat,y])
        # 计算损失函数
        loss = loss_fn(paddle.flatten(yhat), paddle.flatten(y))
        if ind % 50 == 0:
            print("loss:", loss.numpy())
        losses.append(loss.numpy()[0])
        loss.backward()
        # 更新权重
        opt.step()
        opt.clear_grad()
    print("epoch loss:{}".format(np.average(np.array(losses))))
    with open("{}/loss.txt".format(save_dir.split("/")[0]), "a") as f:
        f.write("epoch {} loss:{}\n".format(epoch + 1, np.average(np.array(losses))))
    # paddle.save(opt.state_dict(), "{}/final.pdopt".format(save_dir))

    model.eval()
    TP, FP, FN, TN = 0, 0, 0, 0
    auc_pred = []
    auc_label = []
    # adam.set_state_dict(opt_state_dict)
    for batch_id, data in enumerate(finetune_eval_dataloader):

        x_data = data[0]  # 测试数据
        # x_data = paddle.unsqueeze(x_data,axis=0)
        label = data[1][0].numpy()  # 测试数据标签
        # label h,w
        path = data[2][0]
        predicts = model(x_data)[0].numpy()  # 预测结果
        # pred 1,h,w
        try:
            auc = roc_auc_score(label.flatten(), predicts.flatten())
            if True:
                auc_pred.append(predicts.flatten())
                auc_label.append(label.flatten())
        except ValueError as e:
            print(e)
            print("auc error:", path.split(" ")[-1])
            auc = 1
        # print("auc:", auc)
        predicts = sigmoid(predicts)
        # predicts = paddle.nn.functional.sigmoid(predicts)
        predicts[predicts > 0.5] = 1
        predicts[predicts <= 0.5] = 0

        # acc = paddle.metric.accuracy(predicts, y_data)
        pred = predicts  # [0]

        TP += np.sum(np.array(pred * label))

        FP += np.sum(np.array(pred * (1 - label)))

        FN += np.sum(np.array((1 - pred) * label))

        TN += np.sum(np.array((1 - pred) * (1 - label)))

        # print("{}".format(val_set[ind].replace("data/data152499/mask/","test/casia_eval/pre/"))) #
        # print("{}".format(val_set[ind].replace("data/data163850/gt/Sp/","test/casia_eval/pre/")))

        # print(TP,FP,FN,TN)
        # print(path.split(" ")[0].split("/")[-1], auc, TP / (TP + FP), TP / (TP + FN))

        # cv2.imwrite("{}".format(val_set[ind].replace("data/data163850/gt/Sp/","test/casia_eval/pre/")),cv2.resize(np.float32(pred[0]*255),(384,256)))
        ind += 1
    print("{},{},{},{}".format(TP, FP, FN, TN))
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    print("f1:", f1)
    auc_label=np.array(auc_label).flatten()
    auc_pred=np.array(auc_pred).flatten()
    auc_total = roc_auc_score(auc_label,auc_pred)
    print("auc_total", auc_total)
    stop_count += 1
    if f1 > max_f1:
        stop_count = 0
        max_f1 = f1
        print("max_f1:", max_f1)
        if save_fn!="":
            paddle.save(model.state_dict(), "{}/{}".format(save_dir, save_fn))
        else:
            paddle.save(model.state_dict(), "{}/{}".format(save_dir, file_name.replace("maxf1", "").replace(".pdparams",
                                                                                                            ".finetune.pdparams").replace(
                "san", dataset)))
        with open("{}/f1.txt".format(save_dir.split("/")[0]), "a") as f:
            f.write("max f1:{}\n".format(f1))
    if auc_total > max_auc:
        stop_count = 0
        max_auc = auc_total
        print("max_auc:", max_auc)
        if save_fn!="":
            paddle.save(model.state_dict(), "{}/{}".format(save_dir, save_fn))
        else:
            paddle.save(model.state_dict(), "{}/{}".format(save_dir, file_name.replace("maxf1", "").replace(".pdparams",
                                                                                                            "auc.finetune.pdparams").replace(
                "san", dataset)))
        with open("{}/f1.txt".format(save_dir.split("/")[0]), "a") as f:
            f.write("max auc:{}\n".format(auc_total))
    print("\n")
    with open("{}/f1.txt".format(save_dir.split("/")[0]), "a") as f:
        f.write("epoch {} f1:{}\n".format(epoch + 1, f1))
    if stop and stop_count >= stop_countT:
        with open("{}/loss.txt".format(save_dir.split("/")[0]), "a") as f:
            f.write("stop\n\n")
        with open("{}/f1.txt".format(save_dir.split("/")[0]), "a") as f:
            f.write("stop\n\n")
        break
    if epoch >= 7:
        # break
        pass
    # time.sleep(60)  # 等待缓存释放
# if training:
#     with open("{}/final.txt".format(save_dir),"a") as f:
#         f.write("max val f1:{}\n".format(max_f1))
# save
# paddle.save(model.state_dict(), "{}/{}".format(save_dir,file_name)) if training


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

# opt_state_dict = paddle.load("adam.pdopt")

model.set_state_dict(layer_state_dict)
# model.eval() if index!=7 else model.train()
model.eval()
print("model eval")

# st = "CASIA1" if dataset == "casia1_splice" else "Columbia"
path = "data/test/{}/{}".format(eval_set, file_name.split(".")[1])
if not os.path.exists(path):
    # os.mkdir(path)
    pass
#
# ind = 0
# TP, FP, FN, TN = 0, 0, 0, 0
# # adam.set_state_dict(opt_state_dict)
# AUC = 0
# # model.eval()
# for batch_id, data in enumerate(finetune_eval_dataloader):
#
#     x_data = data[0]  # 测试数据
#     # x_data = paddle.unsqueeze(x_data,axis=0)
#     label = data[1][0].numpy()  # 测试数据标签
#     # label h,w
#     path = data[2][0]
#     predicts = model(x_data)[0].numpy()  # 预测结果
#     # pred 1,h,w
#     try:
#         auc = roc_auc_score(label.flatten(), predicts.flatten())
#     except ValueError as e:
#         print(e)
#         print("auc error:", path.split(" ")[-1])
#         auc = 1
#     # auc = 1
#     AUC += auc
#     # print("auc:", auc)
#     # print(predicts.shape)
#     # 计算损失与精度
#     # loss = loss_fn(predicts, y_data)
#     predicts = sigmoid(predicts)
#
#     # pre = np.copy(predicts)
#     # pre[pre>0.5]=1
#     # pre[pre<=0.5]=0
#     # cv2.imwrite("out.jpg", 255 * pre.transpose((2, 1, 0)))
#     # predicts = erode(predicts)
#
#     # predicts = paddle.nn.functional.sigmoid(predicts)
#     predicts[predicts > 0.5] = 1
#     predicts[predicts <= 0.5] = 0
#     # cv2.imwrite("out_erode.jpg", 255 * predicts.transpose((2, 1, 0)))
#
#     # acc = paddle.metric.accuracy(predicts, y_data)
#     pred = predicts#[0]
#     # print(np.sum(np.float32(pred[0]*255)))
#
#     TP += np.sum(np.array(pred * label))
#
#     FP += np.sum(np.array(pred * (1 - label)))
#
#     FN += np.sum(np.array((1 - pred) * label))
#     # FN偏高，负的推理成正的了
#
#     TN += np.sum(np.array((1 - pred) * (1 - label)))
#
#     print(path.split(" ")[0].split("/")[-1], auc, TP / (TP + FP), TP / (TP + FN))
#     print(TP,FP,FN,TN)
#
#     # path = "data/test/{}/{}/{}.jpg".format(eval_set, file_name.split(".")[1], data[2][0].split(" ")[0][:-4])
#     # cv2.imwrite(path, cv2.resize(np.float32(pred[0] * 255), (384, 256)))
#
#     ind += 1
#     # break
# print("{},{},{},{}".format(TP, FP, FN, TN))
# precision = TP / (TP + FP)
# recall = TP / (TP + FN)
# f1 = 2 * precision * recall / (precision + recall)
# # print("val set size:", len(val_set))
# print("auc", AUC / ind)
# print("precision:{}\nrecall:{}\nf1:{}\n".format(precision, recall, f1))
# with open("{}/f1.txt".format(save_dir.split("/")[0]), "a") as f:
#     f.write("auc:{}\n".format(AUC / ind))
#     f.write("precision:{}\nrecall:{}\nf1:{}\n\n".format(precision, recall, f1))
#
# with open("{}/loss.txt".format(save_dir.split("/")[0]), "a") as f:
#     f.write("\n\n\n")
# os.system("cd test/casia_eval && zip -q -r pre.zip pre")

# os.system("python evaluate.py")