import os
import matplotlib.pyplot as plt
import numpy as np
import math

print(os.getcwd())
def get_path(image_path):
    files=[]
    for dir_name in os.listdir(image_path):
        for image_name in os.listdir(os.path.join(image_path,dir_name)):
            if image_name.endswith('.png') and not image_name.startswith('.'):
                files.append(os.path.join(image_path,dir_name,image_name))
    return sorted(files)

def get_data(mode="train"):
    if not os.path.exists("train_list.txt"):
        data=[]
        for image in os.listdir("data/data152499/tam"):
            image = image.split(".")
            if not os.path.exists("data/data152499/mask/{}".format(image[0]+"_gt.png")):
                continue
            d="data/data152499/tam/{} data/data152499/mask/{}".format(image[0]+".jpg",image[0]+"_gt.png")
            data.append(d)
        length=len(data)
        offset=int(0.9*length)
        if not os.path.exists("data.txt"):
            with open("data.txt","a")as f:
                for d in data:
                    f.write(d+"\n")
        with open("train_list.txt","w")as f:
            for d in data[:offset]:
                f.write(d+"\n")
        with open("val_list.txt","w")as f:
            for d in data[offset:]:
                f.write(d+"\n")
        data=data[:offset] if mode=="train" else data[offset:]
    else:
        l="train_list.txt" if mode=="train" else "val_list.txt"
        data=open(l).readlines()
        data=[d.strip() for d in data]
    return data

def get_san_data(dataset="san",mode="train"):
    l=f"data/train_list_{dataset}.txt" if mode=="train" else "data/val_list_san.txt"
    data=open(l).readlines()
    data=[d.strip() for d in data]
    return data

def get_splice_data(mode="train"):
    l="data/train_list_san.txt" if mode=="train" else "data/val_list_splice.txt"
    data=open(l).readlines()
    data=[d.strip() for d in data]
    return data

def get_dataset(dataset,mode="train",ratio=0.8):
    f="data/list_{}.txt".format(dataset)
    data=open(f).readlines()
    data=[d.strip() for d in data]
    l=int(len(data)*ratio)
    print("data size:{1}/{0}".format(len(data),l))
    ret = data[l:]
    if mode=="test":
        ret.extend(open("data/bak.txt").readlines())
        ret = [r.strip() for r in ret]
    return data[:l] if mode=="train" else ret

def get_imagenet_data(mode="train"):
    if not os.path.exists("train_imagenet_list.txt"):
        data=[]
        for image in os.listdir("seg_train/image"):
            image = image.split(".")
            if not os.path.exists("seg_train/label/{}".format(image[0]+".png")):
                continue
            d="seg_train/image/{} seg_train/label/{}".format(image[0]+".png",image[0]+".png")
            data.append(d)
        length=len(data)
        offset=int(0.9*length)
        if not os.path.exists("data.txt"):
            with open("data.txt","a")as f:
                for d in data:
                    f.write(d+"\n")
        with open("train_imagenet_list.txt","w")as f:
            for d in data[:offset]:
                f.write(d+"\n")
        with open("val_imagenet_list.txt","w")as f:
            for d in data[offset:]:
                f.write(d+"\n")
        data=data[:offset] if mode=="train" else data[offset:]
    else:
        l="train_imagenet_list.txt" if mode=="train" else "val_imagenet_list.txt"
        data=open(l).readlines()
        data=[d.strip() for d in data]
    return data

def show_images(imgs):
    #imgs是一个列表，列表里是多个tensor对象
    #定义总的方框的大小
    plt.figure(figsize=(3*len(imgs),3), dpi=80)
    for i in range(len(imgs)):
        #定义小方框
        plt.subplot(1, len(imgs), i + 1)
        #matplotlib库只能识别numpy类型的数据，tensor无法识别
        imgs[i]=imgs[i].numpy()
        #展示取出的数据
        plt.imshow(imgs[i][0],cmap="gray",aspect="auto")
        #设置坐标轴
        plt.xticks([])
        plt.yticks([])

def label_trans(label):
    label = label.transpose((2, 0, 1))
    label = label[0, :, :]
    label = np.expand_dims(label, axis=0)
    if np.mean(label)>1:
        label=label/255.
    label = label.astype("int64")

def get_test_data(test_images_path):
    test_data=[]
    for name in os.listdir(test_images_path):
        img_path=os.path.join(test_images_path,name)
        test_data.append(img_path)
    test_data=np.expand_dims(np.array(test_data),axis=1)
    return test_data

def erode(img,threshod=0.5):
    _,h,w = img.shape
    ret = np.copy(img)
    # img = sigmoid(img)
    for lh in range(1,h-1):
        for lw in range(1,w-1):
            avg=np.average([img[0,lh,lw],img[0,lh-1,lw-1],img[0,lh-1,lw],\
                           img[0,lh+1,lw+1],img[0,lh+1,lw-1],img[0,lh+1,lw],\
                           img[0,lh-1,lw+1],img[0,lh,lw-1],img[0,lh,lw+1]])
            # print(avg)
            if avg<threshod:
                ret[0,lh,lw] = 0
    return ret

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

# def sigmoid(x):
#     x_ravel = x.ravel()  # 将numpy数组展平
#     length = len(x_ravel)
#     y = []
#     for index in range(length):
#         if x_ravel[index] >= 0:
#             y.append(1.0 / (1 + np.exp(-x_ravel[index])))
#         else:
#             y.append(np.exp(x_ravel[index]) / (np.exp(x_ravel[index]) + 1))
#     return np.array(y).reshape(x.shape)

import cv2
import paddle
# 以正确标签lebal获取gradcam热图
# 获取 Grad-CAM 类激活热图
def get_gradcam(model, data, label, class_dim=2):
    data, label = paddle.to_tensor(data), paddle.to_tensor(label)

    # se_resnet50vd的forward过程
    y = model.conv1_1(data)
    y = model.conv1_2(y)
    y = model.conv1_3(y)
    y = model.pool2d_max(y)
    for block in model.block_list:
        y = block(y)
    conv = y  # 得到模型最后一个卷积层的特征图
    y = model.pool2d_avg(y)
    y = paddle.reshape(y, shape=[-1, model.pool2d_avg_channels])
    predict = model.out(y)  # 得到前向计算的结果

    label = paddle.reshape(label, [-1])
    predict_one_hot = paddle.nn.functional.one_hot(label, class_dim) * predict  # 将模型输出转化为one-hot向量
    score = paddle.mean(predict_one_hot)  # 得到预测结果中概率最高的那个分类的值
    score.backward()  # 反向传播计算梯度
    grad_map = conv.grad  # 得到目标类别的loss对最后一个卷积层输出的特征图的梯度
    grad = paddle.mean(paddle.to_tensor(grad_map), (2, 3), keepdim=True)  # 对特征图的梯度进行GAP（全局平局池化）
    gradcam = paddle.sum(grad * conv, axis=1)  # 将最后一个卷积层输出的特征图乘上从梯度求得权重进行各个通道的加和
    gradcam = paddle.maximum(gradcam, paddle.to_tensor(0.))  # 进行ReLU操作，小于0的值设为0
    for j in range(gradcam.shape[0]):
        gradcam[j] = gradcam[j] / paddle.max(gradcam[j])  # 分别归一化至[0, 1]
    return gradcam


# 将 Grad-CAM 叠加在原图片上显示激活热图的效果
def show_gradcam(model, data, label, class_dim=2, pic_size=256, figsize=None):
    heat_maps = []
    gradcams = get_gradcam(model, data, label)
    for i in range(data.shape[0]):
        img = (data[i] * 255.).astype('uint8').transpose([1, 2, 0])  # 归一化至[0,255]区间，形状：[h,w,c]
        heatmap = cv2.resize(gradcams[i].numpy() * 255., (data.shape[2], data.shape[3])).astype(
            'uint8')  # 调整热图尺寸与图片一致、归一化
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热图转化为“伪彩热图”显示模式
        superimposed_img = cv2.addWeighted(heatmap, .3, img, .7, 0.)  # 将特图叠加到原图片上
        heat_maps.append(superimposed_img)
    heat_maps = np.array(heat_maps)

    heat_maps = heat_maps.reshape([-1, 8, pic_size, pic_size, 3])
    heat_maps = np.concatenate(tuple(heat_maps), axis=1)
    heat_maps = np.concatenate(tuple(heat_maps), axis=1)
    cv2.imwrite('./output/pics/gradcam_label.jpg', heat_maps)

    if figsize != None:
        plt.figure(figsize=figsize, dpi=80)
    heat_maps = plt.imread('./output/pics/gradcam_label.jpg')
    plt.imshow(heat_maps)
    plt.xticks([])
    plt.yticks([])
    plt.show()


# show_gradcam(model, imgs, labels, pic_size=256, figsize=(14, 14))