#coding:utf-8
import argparse
import os
'''
def predict_img(net,
                full_img,
                scale_factor=0.5,
                out_threshold=0.5,
                use_dense_crf=True,
                use_gpu=False):

    net.eval() #进入网络的验证模式，这时网络已经训练好了
    img_height = full_img.size[1] #得到图片的高
    img_width = full_img.size[0] #得到图片的宽

    img = resize_and_crop(full_img, scale=scale_factor) #在utils文件夹的utils.py中定义的函数，重新定义图像大小并进行切割，然后将图像转为数组np.array
    img = normalize(img) #对像素值进行归一化，由[0,255]变为[0,1]

    left_square, right_square = split_img_into_squares(img)#将图像分成左右两块，来分别进行判断

    left_square = hwc_to_chw(left_square) #对图像进行转置，将(H, W, C)变为(C, H, W),便于后面计算
    right_square = hwc_to_chw(right_square)

    X_left = torch.from_numpy(left_square).unsqueeze(0) #将(C, H, W)变为(1, C, H, W)，因为网络中的输入格式第一个还有一个batch_size的值
    X_right = torch.from_numpy(right_square).unsqueeze(0)
    
    if use_gpu:
        X_left = X_left.cuda()
        X_right = X_right.cuda()

    with torch.no_grad(): #不计算梯度
        output_left = net(X_left)
        output_right = net(X_right)

        left_probs = output_left.squeeze(0)
        right_probs = output_right.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(), #重新变成图片
                transforms.Resize(img_height), #恢复原来的大小
                transforms.ToTensor() #然后再变成Tensor格式
            ]
        )
        
        left_probs = tf(left_probs.cpu())
        right_probs = tf(right_probs.cpu())

        left_mask_np = left_probs.squeeze().cpu().numpy()
        right_mask_np = right_probs.squeeze().cpu().numpy()

    full_mask = merge_masks(left_mask_np, right_mask_np, img_width)#将左右两个拆分后的图片合并起来

    #对得到的结果根据设置决定是否进行CRF处理
    if use_dense_crf:
        full_mask = dense_crf(np.array(full_img).astype(np.uint8), full_mask)

    return full_mask > out_threshold



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='MODEL.pth', #指明使用的训练好的模型文件，默认使用MODEL.pth
                        metavar='FILE',
                        help="Specify the file in which is stored the model"
                             " (default : 'MODEL.pth')")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',  #指明要进行预测的图像文件
                        help='filenames of input images', required=True)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+', #指明预测后生成的图像文件的名字
                        help='filenames of ouput images')
    parser.add_argument('--cpu', '-c', action='store_true', #指明使用CPU
                        help="Do not use the cuda version of the net",
                        default=False)
    parser.add_argument('--viz', '-v', action='store_true', 
                        help="Visualize the images as they are processed", #当图像被处理时，将其可视化
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true', #不存储得到的预测图像到某图像文件中，和--viz结合使用，即可对预测结果可视化，但是不存储结果
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--no-crf', '-r', action='store_true', #指明不使用CRF对输出进行后处理
                        help="Do not use dense CRF postprocessing",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float, 
                        help="Minimum probability value to consider a mask pixel white", #最小概率值考虑掩模像素为白色
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images", #输入图像的比例因子
                        default=0.5)

    return parser.parse_args()

def get_output_filenames(args):#从输入的选项args值中得到输出文件名
    in_files = args.input 
    out_files = []

    if not args.output: #如果在选项中没有指定输出的图片文件的名字，那么就会根据输入图片文件名，在其后面添加'_OUT'后缀来作为输出图片文件名
        for f in in_files:
            pathsplit = os.path.splitext(f) #将文件名和扩展名分开，pathsplit[0]是文件名,pathsplit[1]是扩展名
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1])) #得到输出图片文件名
    elif len(in_files) != len(args.output): #如果设置了output名,查看input和output的数量是否相同，即如果input是两张图，那么设置的output也必须是两个，否则报错
        print("Error : Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files

def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8)) #从数组array转成Image

if __name__ == "__main__":
    args = get_args() #得到输入的选项设置的值
    in_files = args.input #得到输入的图像文件
    out_files = get_output_filenames(args) #从输入的选项args值中得到输出文件名

    net = UNet(n_channels=3, n_classes=1) #定义使用的model为UNet，调用在UNet文件夹下定义的unet_model.py,定义图像的通道为3，即彩色图像，判断类型设为1种

    print("Loading model {}".format(args.model)) #指定使用的训练好的model

    if not args.cpu: #指明使用GPU
        print("Using CUDA version of the net, prepare your GPU !")
        net.cuda()
        net.load_state_dict(torch.load(args.model))
    else: #否则使用CPU
        net.cpu()
        net.load_state_dict(torch.load(args.model, map_location='cpu'))
        print("Using CPU version of the net, this may be very slow")

    print("Model loaded !")

    for i, fn in enumerate(in_files): #对图片进行预测
        print("\nPredicting image {} ...".format(fn))

        img = Image.open(fn)
        if img.size[0] < img.size[1]: #(W, H, C)
            print("Error: image height larger than the width")

        mask = predict_img(net=net,
                           full_img=img, #(W, H, C)
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           use_dense_crf= not args.no_crf,
                           use_gpu=not args.cpu)

        if args.viz: #可视化输入的图片和生成的预测图片
            print("Visualizing results for image {}, close to continue ...".format(fn))
            plot_img_and_mask(img, mask)

        if not args.no_save:#设置为False，则保存
            out_fn = out_files[i]
            result = mask_to_image(mask) #从数组array转成Image
            result.save(out_files[i]) #然后保存

            print("Mask saved to {}".format(out_files[i]))
'''

#coding:utf-8
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian

def dense_crf(img, output_probs):
    # img为输入的图像，output_probs是经过网络预测后得到的结果
    # img (W, H, C)
    h = output_probs.shape[0] #高度
    w = output_probs.shape[1] #宽度

    output_probs = np.expand_dims(output_probs, 0)
    output_probs = np.append(1 - output_probs, output_probs, axis=0)

    d = dcrf.DenseCRF2D(w, h, 2) #NLABELS=2两类标注，车和不是车
    U = -np.log(output_probs) #得到一元势
    U = U.reshape((2, -1)) #NLABELS=2两类标注
    U = np.ascontiguousarray(U) #返回一个地址连续的数组
    img = np.ascontiguousarray(img)

    d.setUnaryEnergy(U) #设置一元势

    d.addPairwiseGaussian(sxy=20, compat=3) #设置二元势中高斯情况的值
    d.addPairwiseBilateral(sxy=30, srgb=20, rgbim=img, compat=10)#设置二元势众双边情况的值

    Q = d.inference(5) #迭代5次推理
    Q = np.argmax(np.array(Q), axis=0).reshape((h, w)) #得列中最大值的索引结果

    return Q


def CRFs(image, output_probs):
    # img为输入的图像，output_probs是经过网络预测后得到的结果
    # img (H, W, C)
    img = image

    # 将predicted_image的RGB颜色转换为uint32颜色 0xbbggrr
    anno_rgb = output_probs.transpose((2, 1, 0)).astype(np.uint32)
    anno_lbl = anno_rgb[:, :, 0] + (anno_rgb[:, :, 1] << 8) + (anno_rgb[:, :, 2] << 16)

    # 将uint32颜色转换为1,2,...
    colors, labels = np.unique(anno_lbl, return_inverse=True)

    # 如果你的predicted_image里的黑色（0值）不是待分类类别，表示不确定区域，即将分为其他类别
    # 那么就取消注释以下代码
    # HAS_UNK = 0 in colors
    # if HAS_UNK:
    # colors = colors[1:]

    # 创建从predicted_image到32位整数颜色的映射。
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:, 0] = (colors & 0x0000FF)
    colorize[:, 1] = (colors & 0x00FF00) >> 8
    colorize[:, 2] = (colors & 0xFF0000) >> 16

    # 计算predicted_image中的类数。
    n_labels = len(set(labels.flat))
    # n_labels = len(set(labels.flat)) - int(HAS_UNK) ##如果有不确定区域，用这一行代码替换上一行

    ###########################
    ###     设置CRF模型     ###
    ###########################
    use_2d = False
    use_2d = True
    ###########################################################
    ##不是很清楚什么情况用2D
    ##作者说“对于图像，使用此库的最简单方法是使用DenseCRF2D类”
    ##作者还说“DenseCRF类可用于通用（非二维）密集CRF”
    ##但是根据我的测试结果一般情况用DenseCRF比较对
    #########################################################33
    if use_2d:
        # 使用densecrf2d类
        d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)

        # 得到一元势（负对数概率）
        U = unary_from_labels(labels, n_labels, gt_prob=0.2, zero_unsure=None)
        # U = unary_from_labels(labels, n_labels, gt_prob=0.2, zero_unsure=HAS_UNK)## 如果有不确定区域，用这一行代码替换上一行
        d.setUnaryEnergy(U)

        # 增加了与颜色无关的术语，只是位置-----会惩罚空间上孤立的小块分割,即强制执行空间上更一致的分割
        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)

        # 增加了颜色相关术语，即特征是(x,y,r,g,b)-----使用局部颜色特征来细化它们
        d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=img, compat=10,
                               kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NORMALIZE_SYMMETRIC)
        '''
        addPairwiseGaussian函数里的sxy为公式中的 $\theta_{\gamma}$, 
        addPairwiseBilateral函数里的sxy、srgb为$\theta_{\alpha}$ 和 $\theta_{\beta}$
        '''
    else:
        # 使用densecrf类
        d = dcrf.DenseCRF(img.shape[1] * img.shape[0], n_labels)

        # 得到一元势（负对数概率）
        U = unary_from_labels(labels, n_labels, gt_prob=0.5, zero_unsure=None)
        # U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=HAS_UNK)## 如果有不确定区域，用这一行代码替换上一行
        d.setUnaryEnergy(U)

        # 这将创建与颜色无关的功能，然后将它们添加到CRF中
        feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
        d.addPairwiseEnergy(feats, compat=8, kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

        # 这将创建与颜色相关的功能，然后将它们添加到CRF中
        feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13),
                                          img=img, chdim=2)
        d.addPairwiseEnergy(feats, compat=10,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

    ####################################
    ###         做推理和计算         ###
    ####################################

    # 进行5次推理
    Q = d.inference(10)

    # 找出每个像素最可能的类
    MAP = np.argmax(Q, axis=0)

    # 将predicted_image转换回相应的颜色并保存图像
    MAP = colorize[MAP, :]
    return MAP.reshape(output_probs.shape)