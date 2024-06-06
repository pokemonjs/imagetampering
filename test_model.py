import os

import numpy as np

from rrunet.config import *
from dataset import *

file_name = file_names[index]
model = models[index]
save_dir = f"rrunet_pth/{file_name.split('.')[1]}"
model_path = "{}/{}".format(save_dir, file_name.replace("maxf1", "").replace(".pdparams",
                                                                             ".finetune.pdparams").replace(
    "san", dataset))
model_path=f"{save_dir}/final.dual_ca_se_afusion_rrunet.CASIA.finetune_real561.pdparams"
print(model_path)
layer_state_dict = paddle.load(model_path)
model.set_state_dict(layer_state_dict)
model.eval()

def test(img_path,gt_path,test_model_out='test_model_out'):
    print(img_path,gt_path)

    # 测试单张图片推理
    # img_path = "data/CASIA2/img/Tp_D_CNN_M_N_ani00023_ani00024_10205.jpg"
    # gt_path = "data/CASIA2/gt/Tp_D_CNN_M_N_ani00023_ani00024_10205_gt.png"

    # img_path = r"D:\Pokemonjs\Work\data\test\canonxt_kodakdcs330_sub_11.tif"
    # img_path = r"D:\Pokemonjs\Work\data\test\nikond70_kodakdcs330_sub_11.tif"
    # img_path = r"D:\Pokemonjs\Work\data\test\Tp_D_CRN_M_N_ani10112_ani00100_11648.jpg"
    # gt_path = r"D:\Pokemonjs\Work\data\test\Tp_D_CRN_M_N_ani10112_ani00100_11648_gt.png"

    # img_path = r"D:\Pokemonjs\Work\FAKEDATA\NIST16_Clear\val\images\25.jpg"
    # gt_path = r"D:\Pokemonjs\Work\FAKEDATA\NIST16_Clear\val\mask\25.png"

    # img_path = "FAKEDATA/NIST16/train/images/47.jpg"
    # gt_path = "FAKEDATA/NIST16/train/mask/47.png"

    # img_path = "data/CASIA2/img/Tp_S_NNN_S_B_cha00089_cha00089_10187.jpg"
    # gt_path = "data/CASIA2/gt/Tp_S_NNN_S_B_cha00089_cha00089_10187_gt.png"
    gt_path = None
    image, label = get_single_data(img_path, gt_path)
    # print(np.sum(label.numpy()[0]))
    # cv2.imwrite("test_model_in.jpg", 255 * image.numpy()[0].transpose((1, 2, 0)))
    # cv2.imwrite("test_model_gt.png", label.numpy()[0])
    out = model(image).numpy()[0].transpose((1, 2, 0))
    out = sigmoid(out)
    print(out)
    out = cv2.resize(out, (256, 256))
    out[out > 0.5] = 1
    out[out <= 0.5] = 0

    # label = label.numpy()[0]
    # pred = out
    # TP = np.sum(np.array(pred * label))
    # FP = np.sum(np.array(pred * (1 - label)))
    # FN = np.sum(np.array((1 - pred) * label))
    # TN = np.sum(np.array((1 - pred) * (1 - label)))
    # # auc = roc_auc_score(pred.flatten(), label.flatten())
    # auc = 1
    # print(TP, FP, FN, TN, TP + FP + FN + TN)
    # print(f"{auc} {TP / (TP + FP)} {TP / (TP + FN)}")
    # out = cv2.resize(out, (384, 256))
    # out = cv2.dilate(out, np.ones((5, 5)))
    # out = cv2.erode(out, np.ones((3, 3)))
    # label = cv2.resize(label, (384, 256))
    # out = out * label
    cv2.imwrite(f"test/{test_model_out}.png", 255 * out)

if __name__ == '__main__':
    # for img in os.listdir("data/CASIA2/img/"):
    #     test("data/CASIA2/img/"+img,"data/CASIA2/gt/"+img.replace(".jpg","_gt.png").replace(".tif","_gt.png"),img.replace(".jpg","").replace(".tif",""))
    test("test/Sp_D_NRN_A_nat0028_sec0029_0527.jpg","test_model_gt.png")