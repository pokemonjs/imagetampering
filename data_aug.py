from albumentations.augmentations.transforms import (
    VerticalFlip, HorizontalFlip, JpegCompression, GaussNoise
)
from skimage import io
import os
from tqdm import tqdm
from random import choice,shuffle

path_img = 'data/Columbia/img_aug'
path_mask = 'data/Columbia/gt_aug'
data_file = "data/list_columbia_aug.txt"
img_list = []
gt_list = []

ind = 0
for i in tqdm(os.listdir(path_img)):
    name = i.split('.')[0]
    img = os.path.join(path_img, i)
    fix = "_edgemask"
    mask = os.path.join(path_mask, f'{name}{fix}.png')
    if not os.path.exists(img) or not os.path.exists(mask):
        # print(img,mask)
        continue
    if "_q." in img or "_ver." in img or "_G." in img or "_hor." in img:
        continue
    ind += 1
    img_list.append(img)
    gt_list.append(os.path.join(path_mask, '{}_gt.png'.format(name)))
    image = io.imread(img)
    mask = io.imread(mask)
    io.imsave(os.path.join(path_mask, '{}_gt.png'.format(name)), mask)
    from collections import Counter
    print(Counter(list(mask.flatten())))

    list_quality = [50, 60, 70, 80, 90]
    quality = choice(list_quality)
    io.imsave(os.path.join(path_img, '{}_q.jpg'.format(name)), image, quality=quality)
    io.imsave(os.path.join(path_mask, '{}_q_gt.png'.format(name)), mask)
    img_list.append(os.path.join(path_img, '{}_q.jpg'.format(name)))
    gt_list.append(os.path.join(path_mask, '{}_q_gt.png'.format(name)))
    print(Counter(list(mask.flatten())))

    whatever_data = "my name"

    augmentation = VerticalFlip(p=1.0)
    data = {"image": image, "mask": mask, "whatever_data": whatever_data, "additional": "hello"}
    augmented = augmentation(**data)
    image_ver, mask_ver, whatever_data, additional = augmented["image"], augmented["mask"], augmented["whatever_data"], augmented["additional"]
    io.imsave(os.path.join(path_img, '{}_ver.jpg'.format(name)), image_ver, quality=100)
    io.imsave(os.path.join(path_mask, '{}_ver_gt.png'.format(name)), mask_ver)
    img_list.append(os.path.join(path_img, '{}_ver.jpg'.format(name)))
    gt_list.append(os.path.join(path_mask, '{}_ver_gt.png'.format(name)))
    print(Counter(list(mask_ver.flatten())))

    augmentation = HorizontalFlip(p=1.0)
    data = {"image": image, "mask": mask, "whatever_data": whatever_data, "additional": "hello"}
    augmented = augmentation(**data)
    image_hor, mask_hor, whatever_data, additional = augmented["image"], augmented["mask"], augmented["whatever_data"], augmented["additional"]
    io.imsave(os.path.join(path_img, '{}_hor.jpg'.format(name)), image_hor, quality=100)
    io.imsave(os.path.join(path_mask, '{}_hor_gt.png'.format(name)), mask_hor)
    img_list.append(os.path.join(path_img, '{}_hor.jpg'.format(name)))
    gt_list.append(os.path.join(path_mask, '{}_hor_gt.png'.format(name)))
    print(Counter(list(mask_hor.flatten())))

    # augmentation = GaussNoise(var_limit=(10, 50), p=1.0)
    # data = {"image": image, "mask": mask, "whatever_data": whatever_data, "additional": "hello"}
    # augmented = augmentation(**data)
    # image_g, mask_g, whatever_data, additional = augmented["image"], augmented["mask"], augmented["whatever_data"], augmented["additional"]
    # io.imsave(os.path.join(path_img, '{}_G.jpg'.format(name)), image_g, quality=100)
    # io.imsave(os.path.join(path_mask, '{}_G_gt.png'.format(name)), mask_g)
    # img_list.append(os.path.join(path_img, '{}_G.jpg'.format(name)))
    # gt_list.append(os.path.join(path_mask, '{}_G_gt.png'.format(name)))
    # print(Counter(list(mask_g.flatten())))

print(f"total origin:{ind}")
index = list(range(len(img_list)))
shuffle(index)
with open(data_file,"w") as f:
    for ind in index:
        f.write("{} {}\n".format(img_list[ind],gt_list[ind]))