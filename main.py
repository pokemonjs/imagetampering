#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os,sys
sys.path.append("PaddleSeg")
print(os.getcwd())
from PaddleSeg import paddleseg
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
from metrics import *

paddle.seed(123)
np.random.seed(2021)
random.seed(123)


from rrunet.config import *

# In[30]:
iouw = 1
if len(sys.argv)>1:
    index = int(sys.argv[1])
if len(sys.argv)>2:
    dataset = sys.argv[2]
if len(sys.argv)>3:
    iouw = float(sys.argv[3])

file_name=file_names[index]
model=models[index]
flops = paddle.flops(model, [1,3,128,128], print_detail=True)
#基础API训练

save_dir=f"rrunet_pth/{file_name.split('.')[1]}"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

def shape(tt):
    for t in tt:
        print(t.shape)

lr_base=1e-3
training=True
# training=False
epochs=100 if training else 0
# resume=len(open("training_process.txt").readlines())>0
resume=False
print("resume:", resume)
checkpoint=8
print(save_dir)
print(file_name)
loss_fn = paddle.nn.CrossEntropyLoss(axis=1)#
loss_fn = paddle.nn.BCELoss()
loss_fn = BCEIOULoss(iouw)
lr = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=lr_base,T_max=epochs)
# opt=paddle.optimizer.Momentum(learning_rate=lr,parameters=model.parameters(),weight_decay=1e-2)
opt = paddle.optimizer.Adam(learning_rate=lr,parameters=model.parameters())#,weight_decay=1e-5



# model.set_state_dict(paddle.load("{}/final.pdparams".format(save_dir)))
if resume:
    epochs = epochs - int(open("training_process.txt").readline().strip())
    opt.set_state_dict(paddle.load("{}/final.pdopt".format(save_dir)))

max_f1=0
train_dataloader=DataLoader(MyDataset("train","san"),batch_size=batch_size,shuffle=True)
train_dataloader=DataLoader(MyDataset("train","syn"),batch_size=batch_size,shuffle=True)
#val_list_splice.txt casia1中所有的splice图片
# eval_dataloader=DataLoader(NewDataset("val",eval_set),batch_size=1,shuffle=False)

#写入记录文件
with open("{}/f1.txt".format(save_dir.split("/")[0]),"a") as f:
    f.write("pretrain:{}\n".format(file_name))
    f.write("eval_set:{}\n".format(eval_set))

with open("{}/loss.txt".format(save_dir.split("/")[0]),"a") as f:
    f.write("pretrain:{}\n".format(file_name))
    f.write("eval_set:{}\n".format(eval_set))

for epoch in range(epochs):#-checkpoint-1
    print(time.strftime("%d day %H:%M",time.localtime(time.time())))
    print("start epoch:{}/{}".format(epoch+1,epochs))
    model.train()
    losses=[]
    for ind,data in enumerate(train_dataloader):
        #输入计算输出
        x,y=data
        yhat=model(x)
        yhat=paddle.nn.functional.sigmoid(yhat)
        #shape([x,yhat,y])
        #计算损失函数
        loss=loss_fn(paddle.flatten(yhat),paddle.flatten(y))
        if ind%500 == 0:
            print("loss:",loss.numpy())
        losses.append(loss.numpy()[0])
        loss.backward()
        #更新权重
        opt.step()
        opt.clear_grad()
    print("\nepoch loss:{}\n".format(np.average(np.array(losses))))
    with open("{}/loss.txt".format(save_dir.split("/")[0]),"a")as f:
        f.write("epoch {} loss:{}\n".format(epoch+1,np.average(np.array(losses))))
    with open("training_process.txt","w")as f:
        f.write(str(epoch+1))
    
    paddle.save(model.state_dict(), "{}/{}".format(save_dir,file_name))
    if epoch>=50 and (epoch+1) % 10==0:
        paddle.save(model.state_dict(), "{}/{}".format(save_dir, file_name.replace("san", str(epoch))))

    # # 将该模型及其所有子层设置为预测模式。这只会影响某些模块，如Dropout和BatchNorm
    # model.eval()
    # TP,FP,FN,TN=0,0,0,0
    # # adam.set_state_dict(opt_state_dict)
    # ind=0
    # for batch_id, data in enumerate(eval_dataloader):
    #
    #     x_data = data[0]            # 测试数据
    #     # x_data = paddle.unsqueeze(x_data,axis=0)
    #     label = data[1]            # 测试数据标签
    #     predicts = model(x_data)    # 预测结果
    #     predicts=paddle.nn.functional.sigmoid(predicts)
    #     # print(predicts.shape)
    #     # 计算损失与精度
    #     #loss = loss_fn(predicts, y_data)
    #     predicts[predicts>0.5]=1
    #     predicts[predicts<=0.5]=0
    #     # acc = paddle.metric.accuracy(predicts, y_data)
    #     pred=predicts[0]
    #     # print(ind,pred)
    #     # print(val_set[ind],np.sum(np.array(pred * label)))
    #     # print(np.sum(np.float32(pred[0]*255)))
    #
    #     TP += np.sum(np.array(pred * label))
    #
    #     FP += np.sum(np.array(pred * (1 - label)))
    #
    #     FN += np.sum(np.array((1 - pred) * label))
    #
    #     TN += np.sum(np.array((1 - pred) * (1 - label)))
    #
    #     # print("{}".format(val_set[ind].replace("data/data152499/mask/","test/casia_eval/pre/"))) #
    #     # print("{}".format(val_set[ind].replace("data/data163850/gt/Sp/","test/casia_eval/pre/")))
    #     # print(TP,FP,FN,TN)
    #     # cv2.imwrite("{}".format(val_set[ind].replace("data/data163850/gt/Sp/","test/casia_eval/pre/")),cv2.resize(np.float32(pred[0]*255),(384,256)))
    #     ind+=1
    # precision=TP/(TP+FP)
    # recall=TP/(TP+FN)
    # f1=2*precision*recall/(precision+recall)
    # print("precision:{}\nrecall:{}\nf1:{}\n".format(precision,recall,f1))
    # with open("{}/f1.txt".format(save_dir.split("/")[0]),"a")as f:
    #     f.write("epoch {} f1:{}\n".format(epoch+1,f1))
    # if(f1>max_f1):
    #     max_f1=f1
    #     print("max f1:", max_f1)
    #     with open("{}/f1.txt".format(save_dir.split("/")[0]), "a") as f:
    #         f.write("max f1:{}\n".format(max_f1))
    #     paddle.save(model.state_dict(), "{}/{}".format(save_dir,file_name.replace(".san",".sanmaxf1")))
    #     # paddle.save(opt.state_dict(), "{}/final.pdopt".format(save_dir))


with open("{}/f1.txt".format(save_dir.split("/")[0]),"a") as f:
    f.write("\n".format(file_name))

with open("{}/loss.txt".format(save_dir.split("/")[0]),"a") as f:
    f.write("\n".format(file_name))

#验证集测试
#模型推理 基础API
# load
print(save_dir)
layer_state_dict = paddle.load("{}/{}".format(save_dir,file_name))
# opt_state_dict = paddle.load("adam.pdopt")

model.set_state_dict(layer_state_dict)
model.eval()
print("model eval")

#
# import paddle
# import paddle.nn as nn
# from paddle.io import DataLoader,Dataset
# from utils import get_path,get_data
# #from paddle.vision.transforms import Compose, Resize, ColorJitter,RandomHorizontalFlip,RandomVerticalFlip
# from paddleseg.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip, Resize, Normalize, ResizeStepScaling, RandomDistort
# import cv2
#
# batch_size=1
# # dataset="san"
# # dataset="casia"
# # eval_dataloader=DataLoader(MyDataset("val",dataset),batch_size=batch_size,shuffle=False)
# next(eval_dataloader())
# print("dataset")
#
# val_set=open("data/val_list_san.txt" if dataset=="san" else "data/val_list_splice.txt").readlines()
# val_set=[v.strip().split(" ")[-1] for v in val_set]
# # for v in val_set:
# #     # os.system("cp {} test/casia_eval/gt/".format(v))
# #     cv2.imwrite("test/casia_eval/gt/{}".format(v.split("/")[-1]),cv2.imread(v))
# ind=0
# TP,FP,FN,TN=0,0,0,0
# # adam.set_state_dict(opt_state_dict)
# for batch_id, data in enumerate(eval_dataloader):
#
#     x_data = data[0]            # 测试数据
#     # x_data = paddle.unsqueeze(x_data,axis=0)
#     label = data[1]            # 测试数据标签
#     predicts = model(x_data)    # 预测结果
#     predicts=paddle.nn.functional.sigmoid(predicts)
#     # print(predicts.shape)
#     # 计算损失与精度
#     #loss = loss_fn(predicts, y_data)
#     predicts[predicts>0.5]=1
#     predicts[predicts<=0.5]=0
#     # acc = paddle.metric.accuracy(predicts, y_data)
#     pred=predicts[0]
#     print(val_set[ind],np.sum(np.array(pred * label)))
#     # print(np.sum(np.float32(pred[0]*255)))
#
#     TP += np.sum(np.array(pred * label))
#
#     FP += np.sum(np.array(pred * (1 - label)))
#
#     FN += np.sum(np.array((1 - pred) * label))
#
#     TN += np.sum(np.array((1 - pred) * (1 - label)))
#
#     # print("{}".format(val_set[ind].replace("data/data152499/mask/","test/casia_eval/pre/"))) #
#     # print("{}".format(val_set[ind].replace("data/data163850/gt/Sp/","test/casia_eval/pre/")))
#     # print(TP,FP,FN,TN)
#     cv2.imwrite("{}".format(val_set[ind].replace("data/data163850/gt/Sp/","test/casia_eval/pre/")),cv2.resize(np.float32(pred[0]*255),(384,256)))
#     ind+=1
# precision=TP/(TP+FP)
# recall=TP/(TP+FN)
# f1=2*precision*recall/(precision+recall)
# print("precision:{}\nrecall:{}\nf1:{}\n".format(precision,recall,f1))
# with open("{}/final.txt".format(save_dir),"a") as f:
#     f.write("precision:{}\nrecall:{}\nf1:{}\n".format(precision,recall,f1))
#
#
