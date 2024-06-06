import os
import random
from glob import glob
random.seed(123)

FAKE_ROOT = "D:/Pokemonjs/Work/FAKEDATA"
SAN_ROOT = "D:/Pokemonjs/Work/data"

ratio = 8
fake_data = "NIST16_Clear"
fake_data = "CASIA"
fake_data = "JS_IMD2020.txt"
fake_data = "JS_IMD2020"
fake_data = "JS_COLUMBIA"
fake_data_path = f"D:/Pokemonjs/Work/FAKEDATA/{fake_data}/train/images"
pretrain_data_path = f"{SAN_ROOT}/SAN/train/tam"
labeled_path = f"data/splits/{fake_data}/1-{ratio}/labeled.txt"
unlabeled_path = f"data/splits/{fake_data}/1-{ratio}/unlabeled.txt"
val_path = f"data/splits/{fake_data}/val.txt"
val_data_path = f"{FAKE_ROOT}/{fake_data}/val/images"

def tolabel(string, type=fake_data):
    types={
        "NIST16_Clear": [["images","mask"], [".tif",".png"], [".jpg",".png"]],
        "CASIA": [["images","mask"], [".tif","_gt.png"], [".jpg","_gt.png"]],
        "JS_IMD2020.txt": [["images","mask"], [".tif",".png"], [".jpg",".png"]],
        "JS_IMD2020": [["images","mask"], [".tif","_mask.png"], [".jpg","_mask.png"]],
        "JS_COLUMBIA": [["images","mask"], [".tif","_gt.png"], [".jpg","_gt.png"]],
        "SAN": [["tam","mask"], [".jpg","_mask.png"]]
    }
    for r in types[type]:
        string = string.replace(r[0],r[1])
    return string

# labeled
fake_data_img = glob(f"{fake_data_path}/*")
with open(labeled_path,"w")as f:
    for img_fn in fake_data_img:
        img_fn = img_fn.replace("\\","/")
        string = f"{img_fn}\t{tolabel(img_fn)}\n"
        f.write(string)

# san_data_img = list(glob(f"{pretrain_data_path}/*"))
# # san太多了，取4w先用测试一下
# san_data_img = san_data_img[:len(san_data_img)//4]
# random.shuffle(san_data_img)
# partial = len(san_data_img) // ratio
# with open(labeled_path,"a")as fl:
#     for img_fn in san_data_img[:partial]:
#         img_fn = img_fn.replace("\\","/")
#         string = f"{img_fn}\t{tolabel(img_fn,'SAN')}\n"
#         fl.write(string)

# unlabeled
san_data_img = list(glob(f"{pretrain_data_path}/*"))
random.shuffle(san_data_img)
partial = len(san_data_img) // ratio
with open(unlabeled_path,"w")as flu:
    for img_fn in san_data_img[:partial]:
        img_fn = img_fn.replace("\\","/")
        string = f"{img_fn}\t{tolabel(img_fn,'SAN')}\n"
        flu.write(string)

# val_list
val_data_img = glob(f"{val_data_path}/*")
with open(val_path,"w")as f:
    for img_fn in val_data_img:
        img_fn = img_fn.replace("\\","/")
        string = f"{img_fn}\t{tolabel(img_fn)}\n"
        f.write(string)