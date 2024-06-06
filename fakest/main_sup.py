import datetime
import time

import cv2

from dataset.semi import SemiDataset
from model.semseg.deeplabv2 import DeepLabV2
from model.semseg.deeplabv3plus import DeepLabV3Plus
from model.semseg.pspnet import PSPNet
from utils import count_params, meanIOU, color_map
from metrics import MetricUpdater

import argparse
from copy import deepcopy
import numpy as np
import os
from PIL import Image
import torch
from torch.nn import CrossEntropyLoss, DataParallel
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm


MODE = None

dataset = "COVERAGE_AUG"
# dataset = "CASIA"
# dataset = "JS_IMD2020"
# dataset = "JS_COLUMBIA"
dataset = "NIST16_Clear"
timestr=datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
best_model = None

def sout(string):
    print(string)
    with open(f"record/{dataset}_{timestr}.txt","a")as f:
        f.write(str(string)+"\n")

def parse_args():
    semi_setting = f"{dataset}/1-8"
    parser = argparse.ArgumentParser(description='ST and ST++ Framework')

    # basic settings
    parser.add_argument('--data-root', type=str, default="")
    parser.add_argument('--dataset', type=str, default=dataset)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--crop-size', type=int, default=None)
    parser.add_argument('--backbone', type=str, choices=['resnet50', 'resnet101'], default='resnet50')
    parser.add_argument('--model', type=str, choices=['deeplabv3plus', 'pspnet', 'deeplabv2'],
                        default='deeplabv3plus')

    # semi-supervised settings
    parser.add_argument('--labeled-id-path', type=str, default=f"dataset/splits/{semi_setting}/labeled.txt")
    parser.add_argument('--unlabeled-id-path', type=str, default=f"dataset/splits/{semi_setting}/unlabeled.txt")
    parser.add_argument('--pseudo-mask-path', type=str, default=f"outdir/pseudo_masks/{semi_setting}")

    parser.add_argument('--save-path', type=str, default=f"outdir/models/{semi_setting}")

    # arguments for ST++
    parser.add_argument('--reliable-id-path', type=str, default=f"dataset/splits/{semi_setting}")
    parser.add_argument('--plus', dest='plus', default=True, action='store_true',
                        help='whether to use ST++')

    args = parser.parse_args()
    return args


def main(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists(args.pseudo_mask_path):
        os.makedirs(args.pseudo_mask_path)
    if args.plus and args.reliable_id_path is None:
        exit('Please specify reliable-id-path in ST++.')

    criterion = CrossEntropyLoss(ignore_index=255)

    valset = SemiDataset(args.dataset, args.data_root, 'val', None)
    valloader = DataLoader(valset, batch_size=4 if args.dataset == 'cityscapes' else 1,
                           shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

    # <====================== Supervised training with labeled images (SupOnly) ======================>
    sout('\n================> Total stage 1/%i: '
          'Supervised training on labeled images (SupOnly)' % (6 if args.plus else 3))

    global best_model
    global MODE
    MODE = 'train'

    trainset = SemiDataset(args.dataset, args.data_root, MODE, args.crop_size, args.labeled_id_path)
    trainset.ids = 2 * trainset.ids if len(trainset.ids) < 200 else trainset.ids
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                             pin_memory=True, num_workers=16, drop_last=True)

    model, optimizer = init_basic_elems(args)
    sout('\nParams: %.1fM' % count_params(model))

    # best_model = model
    # state_dict = torch.load("/root/ST-PlusPlus-master/outdir/models/CASIA/1-8/deeplabv3plus_resnet50_36.52.pth")
    # print(state_dict.keys())
    # best_model.module.load_state_dict(state_dict)

    best_model, checkpoints = train(model, trainloader, valloader, criterion, optimizer, args)



def init_basic_elems(args):
    model_zoo = {'deeplabv3plus': DeepLabV3Plus, 'pspnet': PSPNet, 'deeplabv2': DeepLabV2}
    # pj4fix,修改类型
    # model = model_zoo[args.model](args.backbone, 21 if args.dataset == 'pascal' else 19)
    model = model_zoo[args.model](args.backbone, 2)

    head_lr_multiple = 10.0
    if args.model == 'deeplabv2':
        assert args.backbone == 'resnet101'
        model.load_state_dict(torch.load('pretrained/deeplabv2_resnet101_coco_pretrained.pth'))
        head_lr_multiple = 1.0

    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': args.lr},
                     {'params': [param for name, param in model.named_parameters()
                                 if 'backbone' not in name],
                      'lr': args.lr * head_lr_multiple}],
                    lr=args.lr, momentum=0.9, weight_decay=1e-4)

    model = DataParallel(model).cuda()

    return model, optimizer


def train(model, trainloader, valloader, criterion, optimizer, args):
    iters = 0
    total_iters = len(trainloader) * args.epochs

    previous_best = 0.0
    # best_model = deepcopy(model)

    global MODE
    global best_model

    if MODE == 'train':
        checkpoints = []

    drop = 0
    for epoch in range(args.epochs):
        sout("\n==> Epoch %i, learning rate = %.4f\t\t\t\t\t previous best = %.2f" %
              (epoch, optimizer.param_groups[0]["lr"], previous_best))

        model.train()
        total_loss = 0.0
        tbar = tqdm(trainloader)

        for i, (img, mask) in enumerate(tbar):
            img, mask = img.cuda(), mask.cuda()

            pred = model(img)
            # print(pred.shape, mask.shape)
            loss = criterion(pred, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            iters += 1
            lr = args.lr * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * 1.0 if args.model == 'deeplabv2' else lr * 10.0

            tbar.set_description('Loss: %.3f' % (total_loss / (i + 1)))

        # pj4fix
        # metric = meanIOU(num_classes=21 if args.dataset == 'pascal' else 19)
        metric = meanIOU(num_classes=2)
        # pj4fix
        f1auc = MetricUpdater()

        model.eval()
        tbar = tqdm(valloader)

        with torch.no_grad():
            for img, mask, _ in tbar:
                img = img.cuda()
                pred = model(img)
                pred = torch.argmax(pred, dim=1)

                metric.add_batch(pred.cpu().numpy(), mask.numpy())
                f1auc.update_f1(pred.cpu().numpy(), mask.numpy())
                mIOU = metric.evaluate()[-1]

                tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))

        f1, auc = f1auc.caculate()
        mIOU = f1 * 100
        sout("F1:%f, AUC:%f" % (f1*100, auc*100))
        # mIOU *= 100.0
        # sout(str(mIOU)+str(previous_best))
        if mIOU >= previous_best:
            # if previous_best != 0:
            #     os.remove(os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, previous_best)))
            previous_best = mIOU
            torch.save(model.module.state_dict(),
                       os.path.join(args.save_path, '%s_%s_sup.pth' % (args.model, args.backbone)))
            sout("new_best_model")
            best_model = deepcopy(model)
            drop = 0
        else:
            drop += 1
        if drop>=10:
            break

        if MODE == 'train' and ((epoch + 1) in [args.epochs // 3, args.epochs * 2 // 3, args.epochs]):
            checkpoints.append(deepcopy(model))

    if MODE == 'train':
        return best_model, checkpoints

    return best_model


def select_reliable(models, dataloader, args):
    if not os.path.exists(args.reliable_id_path):
        os.makedirs(args.reliable_id_path)

    for i in range(len(models)):
        models[i].eval()
    tbar = tqdm(dataloader)

    id_to_reliability = []

    with torch.no_grad():
        for img, mask, id in tbar:
            img = img.cuda()

            preds = []
            for model in models:
                preds.append(torch.argmax(model(img), dim=1).cpu().numpy())

            mIOU = []
            for i in range(len(preds) - 1):
                # pj4fix
                # metric = meanIOU(num_classes=21 if args.dataset == 'pascal' else 19)
                metric = meanIOU(num_classes=2)
                metric.add_batch(preds[i], preds[-1])
                # pj4fix
                f1auc = MetricUpdater()
                f1auc.update_f1(preds[i], preds[-1])
                f1, auc = f1auc.caculate()
                # mIOU.append(metric.evaluate()[-1])
                mIOU.append(f1)

            reliability = sum(mIOU) / len(mIOU)
            id_to_reliability.append((id[0], reliability))

    id_to_reliability.sort(key=lambda elem: elem[1], reverse=True)
    with open(os.path.join(args.reliable_id_path, 'reliable_ids.txt'), 'w') as f:
        for elem in id_to_reliability[:len(id_to_reliability) // 2]:
            f.write(elem[0] + '\n')
    with open(os.path.join(args.reliable_id_path, 'unreliable_ids.txt'), 'w') as f:
        for elem in id_to_reliability[len(id_to_reliability) // 2:]:
            f.write(elem[0] + '\n')


def label(model, dataloader, args):
    model.eval()
    tbar = tqdm(dataloader)

    # pj4fix
    # metric = meanIOU(num_classes=21 if args.dataset == 'pascal' else 19)
    metric = meanIOU(num_classes=2)
    f1auc = MetricUpdater()
    cmap = color_map(args.dataset)

    with torch.no_grad():
        for img, mask, id in tbar:
            img = img.cuda()
            pred = model(img, True)
            pred = torch.argmax(pred, dim=1).cpu()

            metric.add_batch(pred.numpy(), mask.numpy())
            # pj4fix
            f1auc.update_f1(pred.numpy(), mask.numpy())
            # mIOU = metric.evaluate()[-1]
            f1, auc = f1auc.caculate()
            mIOU = f1

            pred = Image.fromarray(pred.squeeze(0).numpy().astype(np.uint8), mode='P')
            pred.putpalette(cmap)

            # pj4fix \t
            path = '%s/%s' % (args.pseudo_mask_path, os.path.basename(id[0].split('\t')[1]))
            pred.save(path)
            # pj4fix
            pred = cv2.imread(path)
            kernel = np.ones((3, 3), np.uint8)
            pred = cv2.dilate(pred, kernel, iterations=10)
            cv2.imwrite(path,pred)

            tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))


if __name__ == '__main__':
    args = parse_args()

    if args.epochs is None:
        args.epochs = 30  # {'pascal': 80, 'cityscapes': 240, 'fake':80}[args.dataset]
    if args.lr is None:
        args.lr = 0.001  # {'pascal': 0.001, 'cityscapes': 0.004, 'fake':0.001}[args.dataset] / 16 * args.batch_size
    if args.crop_size is None:
        args.crop_size = 256  # {'pascal': 321, 'cityscapes': 721, 'fake':321}[args.dataset]

    sout("")
    sout(args)

    main(args)
