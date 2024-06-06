import os.path
import random
import sys
# import imageio
import imageio.v2 as imageio

import numpy as np

sys.path.append("PaddleSeg")


def crop(img, ran=0):
    h = img.shape[0]
    # print(h,ran)
    if ran == 0:
        return img[:, :h]
    else:
        return img[:, -h:]


# crop(cv2.imread("Tp_input.jpg",0)).shape

# 构建数据集 paddleseg数据增强
# 输入原始图像
import paddle
import paddle.nn as nn
from paddle.io import DataLoader, Dataset
from utils import *
from utils import get_path, get_data, get_imagenet_data, get_san_data, get_splice_data
# from paddle.vision.transforms import Compose, Resize, ColorJitter,RandomHorizontalFlip,RandomVerticalFlip
from paddleseg.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip, Resize, Normalize, \
    ResizeStepScaling, RandomDistort
import cv2

size = 256

class MyDataset(Dataset):

    def __init__(self, mode="train", data_set="san"):
        super().__init__()
        self.path = get_san_data(data_set, mode)  # if data_set=="san" else get_splice_data(mode)
        # self.path=get_data(mode)
        # self.path=get_imagenet_data(mode)
        # print(self.path)
        self.leng = len(self.path)
        self.mode = mode
        print(data_set, self.leng)
        if mode == "train":
            # self.transform=Compose([RandomVerticalFlip(0.5),RandomHorizontalFlip(0.5)])#,RandomDistort()
            self.transform = Compose([])
        else:
            self.transform = Compose([])
        # self.transform=None
        if mode == "val":
            for p in self.path:
                # print(p)
                pass

    def __getitem__(self, index):
        p = self.path[index]
        image, label = p.split(" ")
        # image=cv2.imread(image.strip())
        if not os.path.exists(image) or not os.path.exists(label):
            print("not found:", image, label)
        image = cv2.imread(image.strip()).astype("float32")
        label = cv2.imread(label.strip(), 0)  # .replace(".png","_gt.png")
        # ran=random.randint(0,1)#[0,1]
        # if self.mode=="train":
        #     ran = index%2
        #     image=crop(image,ran)
        #     label=crop(label,ran)
        image = (image) / 255.
        image = cv2.resize(image, (size, size))

        # mean, std = [0.4123, 0.4481, 0.4420], [0.2862, 0.2553, 0.2642] # SAN
        mean, std = [0.40343238, 0.44579795, 0.46825019], [0.27494126, 0.25837288, 0.26313265] # SYN
        for i in range(3):
            image[:,:,i] = (image[:,:,i] - mean[i]) / std [i]

        label = cv2.resize(label, (size, size), interpolation=cv2.INTER_NEAREST)
        # label=label.reshape((1,256,256))#.astype("int64")
        label[label > 1] = 1
        if self.transform is not None:
            image, label = self.transform(image, label)  #
            # image=image.transpose((2,0,1))
            # label=self.transform(label).astype(np.int64)
            label = label.astype(np.float32)
        else:
            image = image.transpose((2, 0, 1))
        if self.transform is not None:
            pass
            # cv2.imwrite("dataset_vis/image/{}.jpg".format(index),(image*127.5+127.5).transpose((1,2,0)))#
        else:
            pass
            # cv2.imwrite("dataset_vis/image/{}.jpg".format(index),(image*127.5+127.5).transpose((1,2,0)))
        # cv2.imwrite("dataset_vis/label/{}.png".format(index),(label*255))
        return image, label

    def __len__(self):
        return self.leng


# batch_size = 16 // 4
# batch_size = 16
# train_dataloader = DataLoader(MyDataset("train"), batch_size=batch_size, shuffle=True)
# eval_dataloader = DataLoader(MyDataset("val"), batch_size=batch_size, shuffle=False)
# next(train_dataloader())
# print("dataset")

# In[4]:


# finetune数据集

# 构建数据集 paddleseg数据增强
# 输入原始图像
import paddle
import paddle.nn as nn
from paddle.io import DataLoader, Dataset
from utils import *
from utils import get_path, get_data, get_imagenet_data, get_san_data, get_splice_data
# from paddle.vision.transforms import Compose, Resize, ColorJitter,RandomHorizontalFlip,RandomVerticalFlip
from paddleseg.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip, Resize, Normalize, \
    ResizeStepScaling, RandomDistort
import cv2


class FineTuneDataset(Dataset):

    def __init__(self, mode="train", data_set="casia"):
        super().__init__()
        path = open("finetune_{}_{}.txt".format(data_set, mode)).readlines()
        self.path = [p.strip() for p in path]
        print("data size:", len(self.path))
        self.leng = len(self.path)
        self.mode = mode
        if mode == "train":
            # self.transform=Compose([RandomVerticalFlip(0.5),RandomHorizontalFlip(0.5)])#,RandomDistort()
            self.transform = Compose([])
        else:
            self.transform = Compose([])
        # self.transform=None
        if mode == "val":
            for p in self.path:
                # print(p)
                pass

    def __getitem__(self, index):
        p = self.path[index]
        image, label = p.split(" ")
        # if self.mode == "val":
        #     print(label)
        # image=cv2.imread(image.strip())
        image = cv2.imread(image.strip()).astype("float32")
        label = cv2.imread(label.strip(), 0)  # .replace(".png","_gt.png")
        # image=cv2.resize(image,(256,256))
        # label=cv2.resize(label,(256,256))
        # ran=random.randint(0,1)#[0,1]
        if self.mode == "train":
            ran = index % 2
            image = crop(image, ran)
            label = crop(label, ran)
        image = (image) / 255.
        image = cv2.resize(image, (size, size))
        label = cv2.resize(label, (size, size), interpolation=cv2.INTER_NEAREST)
        # label=label.reshape((1,256,256))#.astype("int64")
        label[label > 1] = 1
        if self.transform is not None:
            image, label = self.transform(image, label)  #
            # image=image.transpose((2,0,1))
            # label=self.transform(label).astype(np.int64)
            label = label.astype(np.float32)
        else:
            image = image.transpose((2, 0, 1))
        if self.transform is not None:
            pass
            # cv2.imwrite("dataset_vis/image/{}.jpg".format(index),(image*127.5+127.5).transpose((1,2,0)))#
        else:
            pass
            # cv2.imwrite("dataset_vis/image/{}.jpg".format(index),(image*127.5+127.5).transpose((1,2,0)))
        # cv2.imwrite("dataset_vis/label/{}.png".format(index),(label*255))
        return image, label

    def __len__(self):
        return self.leng


# batch_size = 16 // 4
# batch_size = 16
# # finetune_train_dataloader=DataLoader(FineTuneDataset("train"),batch_size=batch_size,shuffle=True)
# # finetune_eval_dataloader=DataLoader(FineTuneDataset("val"),batch_size=batch_size,shuffle=False)
# next(train_dataloader())
# print("dataset")

# In[5]:


# 只有一个list的数据集

# 构建数据集 paddleseg数据增强
# 输入原始图像
import paddle
import paddle.nn as nn
from paddle.io import DataLoader, Dataset
from utils import *
from utils import get_path, get_data, get_imagenet_data, get_san_data, get_splice_data
# from paddle.vision.transforms import Compose, Resize, ColorJitter,RandomHorizontalFlip,RandomVerticalFlip
from paddleseg.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip, Resize, Normalize, \
    ResizeStepScaling, RandomDistort
import cv2


class NewDataset(Dataset):

    def __init__(self, mode="train", dataset="casia", ratio=0.8):
        super().__init__()
        path = get_dataset(dataset, mode, ratio)
        self.path = [p.strip() for p in path]
        print("data size:", len(self.path))
        self.leng = len(self.path)
        self.mode = mode
        if mode == "train":
            # self.transform=Compose([RandomVerticalFlip(0.5),RandomHorizontalFlip(0.5)])#,RandomDistort()
            self.transform = Compose([])
        else:
            self.transform = Compose([])
        # self.transform=None
        if mode == "val":
            for p in self.path:
                # print(p)
                pass

    def __getitem__(self, index):
        p = self.path[index]
        image, label = p.split(" ")
        # if self.mode == "val":
        #     print(label)
        # image=cv2.imread(image.strip())
        # print(image,"\n",label)
        # print(image,label)
        if not os.path.exists(image) or not os.path.exists(label):
            print("not found:", image, label)
        image = cv2.imread(image.strip()).astype("float32")
        label = cv2.imread(label.strip(), 0)  # .replace(".png","_gt.png")
        # image=cv2.resize(image,(256,256))
        # label=cv2.resize(label,(256,256))
        # ran=random.randint(0,1)#[0,1]
        # if self.mode=="train":
        #     ran = index%2
        #     image=crop(image,ran)
        #     label=crop(label,ran)

        image = (image) / 255.
        image = cv2.resize(image, (size, size))
        label = cv2.resize(label, (size, size), interpolation=cv2.INTER_NEAREST)
        # label=label.reshape((1,256,256))#.astype("int64")
        label[label > 1] = 1
        # print(self.transform)
        if self.transform is not None:
            image, label = self.transform(image, label)  #
            # image=image.transpose((2,0,1))
            # label=self.transform(label).astype(np.int64)
            label = label.astype(np.float32)
        else:
            image = image.transpose((2, 0, 1))
        if self.transform is not None:
            pass
            # cv2.imwrite("dataset_vis/image/{}.jpg".format(index),(image*127.5+127.5).transpose((1,2,0)))#
        else:
            pass
            # cv2.imwrite("dataset_vis/image/{}.jpg".format(index),(image*127.5+127.5).transpose((1,2,0)))
        # cv2.imwrite("dataset_vis/label/{}.png".format(index),(label*255))
        return image, label, self.path[index]

    def __len__(self):
        return self.leng


# batch_size = 16 // 4
# batch_size = 16
# dataset = "casia"
# # finetune_train_dataloader=DataLoader(NewDataset("train",dataset),batch_size=batch_size,shuffle=True)
# # finetune_eval_dataloader=DataLoader(NewDataset("val",dataset),batch_size=batch_size,shuffle=False)
# next(train_dataloader())
# print("dataset")


class TestDataset(Dataset):

    def __init__(self, mode="train", dataset="casia", ratio=0.8):
        super().__init__()
        path = get_dataset(dataset, mode, 1.0)
        self.path = [p.strip() for p in path]
        print("data size:", len(self.path))
        self.leng = len(self.path)
        self.mode = mode
        if mode == "train":
            # self.transform = Compose([RandomVerticalFlip(0.5), RandomHorizontalFlip(0.5)])  # ,RandomDistort()
            self.transform = Compose([])
        else:
            self.transform = Compose([])
        # self.transform=None
        if mode == "val":
            for p in self.path:
                # print(p)
                pass

    def __getitem__(self, index):
        p = self.path[index]
        image, label = p.split(" ")
        # if self.mode == "val":
        #     print(label)
        # image=cv2.imread(image.strip())
        # print(image,"\n",label)
        # print(image,label)
        if not os.path.exists(image):
            print(image)
        image = cv2.imread(image.strip()).astype("float32")
        label = cv2.imread(label.strip(), 0)  # .replace(".png","_gt.png")
        # image=cv2.resize(image,(256,256))
        # label=cv2.resize(label,(256,256))
        # ran=random.randint(0,1)#[0,1]
        # if self.mode=="train":
        #     ran = index%2
        #     image=crop(image,ran)
        #     label=crop(label,ran)

        image = (image) / 255.
        image = cv2.resize(image, (size, size))
        label = cv2.resize(label, (size, size), interpolation=cv2.INTER_NEAREST)
        # label=label.reshape((1,256,256))#.astype("int64")
        label[label > 1] = 1
        # print(self.transform)
        if self.transform is not None:
            image, label = self.transform(image, label)  #
            # image=image.transpose((2,0,1))
            # label=self.transform(label).astype(np.int64)
            label = label.astype(np.float32)
        else:
            image = image.transpose((2, 0, 1))
        if self.transform is not None:
            pass
            # cv2.imwrite("dataset_vis/image/{}.jpg".format(index),(image*127.5+127.5).transpose((1,2,0)))#
        else:
            pass
            # cv2.imwrite("dataset_vis/image/{}.jpg".format(index),(image*127.5+127.5).transpose((1,2,0)))
        # cv2.imwrite("dataset_vis/label/{}.png".format(index),(label*255))
        return image, label, self.path[index]

    def __len__(self):
        return self.leng


class FinetuneDataset(Dataset):

    def __init__(self, mode="train", dataset="casia", ratio=0.8):
        super().__init__()
        path = f"E:/StudyWork/python/FAKEDATA/{dataset}/{mode}/images"
        fix = "_gt" if "casia" in dataset.lower() or "columbia" in dataset.lower() else ""
        fixes = {"CASIA": "_gt", "JS_COLUMBIA": "_gt", "NIST16": "", "NIST160": "", "NIST16_Clear": "", "JS_COLUMBIA_ORI": "_gt",
                 "COVERAGE": "", "COVERAGE_AUG": "", "JS_COVERAGE": "", "JS_COVERAGE_AUG": "", "JS_IMD2020": "_mask"
                 , "JS_IMD2020_LARGE": "_mask"}
        fix = fixes[dataset]
        self.path = [f"{path}/{p} {path.replace('images', 'mask')}/{p[:-4]}{fix}.png" for p in os.listdir(path)]

        norm = {
            # 'CASIA': [[0.39225175, 0.43358087, 0.44139242], [0.28549757, 0.26002468, 0.26558968]], # from HH
            # 'CASIA': [[0.41249460446624175, 0.45201789799748404, 0.4443054839372888], [0.30117378471556383, 0.267559827708473, 0.27481340088858824]], # 5123
            'CASIA': [[0.4120610556629078, 0.4517823439128274, 0.44400781710237003], [0.30146493171747885, 0.26745353501154745, 0.2747617919727412]], # from HH 重新计算
            'JS_COLUMBIA': [[0.38060766509939237, 0.40590406801189827, 0.3958182434942531], [0.22227525782137647, 0.22297643408326853, 0.23439802218626077]],
            'JS_COLUMBIA_ORI': [[0.38060766509939237, 0.40590406801189827, 0.3958182434942531], [0.22227525782137647, 0.22297643408326853, 0.23439802218626077]],
            'COVERAGE_AUG': [[0.39062718031625715, 0.4547705603693665, 0.49570572256280787], [0.25113356248737856, 0.25659752163392463, 0.26667642841273576]],
            'JS_IMD2020': [[0.4091354536368938, 0.434008737006763, 0.458265430359303], [0.2952756259324592, 0.28488178904672434, 0.29231307819124497]],
            'JS_IMD2020_LARGE': [[0.4028449745157341, 0.42802989254381485, 0.43922090029159727], [0.2927919063428918, 0.2747368055107117, 0.2780600413884747]],
            'NIST16_Clear': [[0.46033082895869915, 0.446666137096956, 0.45626814115823244], [0.28091311977145134, 0.2292632263447054, 0.2489062152532568]]
        }
        self.norm = norm[dataset]
        random.shuffle(self.path)
        print("data size:", len(self.path))
        self.leng = len(self.path)
        self.mode = mode
        if mode == "train":
            # self.transform=Compose([RandomVerticalFlip(0.5),RandomHorizontalFlip(0.5)])#,RandomDistort()
            self.transform = Compose([])
        else:
            self.transform = Compose([])
        self.transform=None
        if mode == "val":
            for p in self.path:
                # print(p)
                pass

    def __getitem__(self, index):
        p = self.path[index]
        image, label = p.split(" ")
        # if self.mode == "val":
        #     print(label)
        # image=cv2.imread(image.strip())
        # print(image,"\n",label)
        # print(image,label)
        if not os.path.exists(image) or not os.path.exists(label):
            print("not found:", image, label)
        # test
        from PIL import Image
        image = Image.open(image)
        image.save("pic.jpg")
        image = cv2.imread("pic.jpg").astype("float32")

        # image = cv2.imread(image.strip()).astype("float32") # BRG
        # print(image.shape)
        label = cv2.imread(label.strip(), 0)  # .replace(".png","_gt.png")
        # image=cv2.resize(image,(256,256))
        # label=cv2.resize(label,(256,256))
        # ran=random.randint(0,1)#[0,1]
        # if self.mode=="train":
        #     ran = index%2
        #     image=crop(image,ran)
        #     label=crop(label,ran)

        image = (image) / 255.
        image = cv2.resize(image, (size, size))

        mean, std = self.norm
        for i in range(3):
            image[:,:,i] = (image[:,:,i] - mean[i]) / std [i]

        # print(image.shape) # hwc
        label = cv2.resize(label, (size, size), interpolation=cv2.INTER_NEAREST)
        # label=label.reshape((1,256,256))#.astype("int64")
        label[label > 1] = 1
        # print(self.transform)
        if self.transform is not None:
            image, label = self.transform(image, label)  #
            # image=image.transpose((2,0,1))
            # label=self.transform(label).astype(np.int64)
            label = label.astype(np.float32)
            # print(image.shape) # hwc
        else:
            image = image.transpose((2, 0, 1))
            label = label.astype(np.float32)
        if self.transform is not None:
            pass
            # cv2.imwrite("dataset_vis/image/{}.jpg".format(index),(image*127.5+127.5).transpose((1,2,0)))#
        else:
            pass
            # cv2.imwrite("dataset_vis/image/{}.jpg".format(index),(image*127.5+127.5).transpose((1,2,0)))
        # cv2.imwrite("dataset_vis/label/{}.png".format(index),(label*255))
        # 返回均为hwc
        return image, label, self.path[index]

    def __len__(self):
        return self.leng

    def get_path(self):
        return self.path


def get_single_data(img_path, gt_path=None):
    image = cv2.imread(img_path).astype("float32")
    image = cv2.resize(image, (size, size)).transpose((2, 0, 1))
    image = image.reshape((1, -1, size, size))
    image = (image) / 255.
    # cv2.imwrite("test_model_in0.jpg", 255 * image[0].transpose((1, 2, 0)))

    if not gt_path is None:
        label = cv2.imread(gt_path.strip(), 0)
        label = cv2.resize(label, (size, size), interpolation=cv2.INTER_NEAREST)
        label = label.reshape((-1, size, size))
    else:
        label = np.ones([1, size, size])
    # image, label = transform(image, label)
    image = paddle.to_tensor(image)
    label = paddle.to_tensor(label)
    # cv2.imwrite("test_model_in1.jpg", 255 * image.numpy()[0].transpose((1, 2, 0)))
    print(image.shape, label.shape)
    return image, label

