# 导入相关的包
import os
import sys

import cv2
import numpy as np
import torch
import torch.optim
from PIL import Image
from skimage.morphology import remove_small_objects, binary_opening
from torchvision import transforms


# 将图片转为二值图
def transform2img(base_path, save_path):
    for im in os.listdir(base_path):
        img = cv2.imread(os.path.join(base_path, im))
        b, g, r = cv2.split(img)
        # print(np.unique(img))
        b[np.where(b != 0)] = 1
        # ret, mask = cv2.threshold(img2gray, 128, 255, cv2.THRESH_BINAR)
        # print(b)
        cv2.imwrite(os.path.join(save_path, im), b)


# 获取路径下的所有文件
def get_imlist(dirPath):
    """
    :param dirPath: 文件夹路径
    :return:
    """
    dst = []
    # 对目录下的文件进行遍历
    for file in os.listdir(dirPath):
        # 判断是否是文件
        if os.path.isfile(os.path.join(dirPath, file)):
            c = os.path.basename(file)
            print(c)
            name = dirPath + '/' + c

            img = Image.open(name)
            img = img.resize((3072, 2560), Image.ANTIALIAS)
            img = np.asarray(img, dtype=np.int32)
            # img = cv2.imread(name, cv2.IMREAD_UNCHANGED)
            # img = cv2.resize(img, (3072, 2560))  # 使尺寸大小一样
            # img = cv2.resize(img, (2560, 3072))  # 使尺寸大小一样
            # img = torch.tensor(img).float()
            print(img.shape)
            dst.append(img)
    return dst


# 获取mask路径下所有的mask
def mask_get_imlist(dirPath):
    """
    :param dirPath: 文件夹路径
    :return:
    """
    dst = []
    # 对目录下的文件进行遍历
    for file in os.listdir(dirPath):
        # 判断是否是文件
        if os.path.isfile(os.path.join(dirPath, file)):
            c = os.path.basename(file)
            print(c)
            name = dirPath + '/' + c

            # print(Image.open(name).mode)
            img = cv2.imread(name, cv2.IMREAD_UNCHANGED)
            print(img.ndim)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
            # 二值化
            ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY);
            cv2.imwrite('./Ti/mask_data/' + c, img)

            # img = torch.tensor(img).float()
            print(img.shape)
            dst.append(img)
    return dst


# 将tensor转为image并保存在当地文件夹
def mask_save_tensor_img(imgs, path, type):
    print(imgs.shape)
    unloader = transforms.ToPILImage()
    b, h, w, c = imgs.shape

    for i, img in enumerate(imgs):
        # print(i)
        img = imgs[i]

        # print(img.shape)
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)
        pic = unloader(img)
        pic.save(path + str(i) + '_r.' + type)
    # plt.imshow(img)


# 将tensor转为image并保存在当地文件夹
def save_tensor_img(imgs, path, type):
    print(imgs.shape)
    unloader = transforms.ToPILImage()
    b, h, w, c = imgs.shape

    for i, img in enumerate(imgs):
        # print(i)
        img = imgs[i]

        # print(img.shape)
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)
        pic = unloader(img)
        pic.save(path + str(i) + '_r.' + type)
    # plt.imshow(img)


# In[5]:


# 对图片进行边缘填充
def img_padding(imgs, patch_size: tuple, stride_size: tuple):
    """从上到下，从左到右依次对图像进行切片，每个图像块大小相同，相互之间间隔相等
            Args:
                imgs:单张/一个批次的图像，(h,w,c)/(b,h,w,c);
                patch_size: 图像块的大小，（patch_h,patch_w）
                stride_size:每个图像块的间隔，（stride_h,strde_w）

            returns:
                numpy,array:(n_patches,patch_h,patch_w)
        """
    # 判断图片的维数
    assert imgs.ndim > 2

    if imgs.ndim == 3:
        # (h,w,c)->(1,h,w,c)
        imgs = np.expand_dims(imgs, axis=0)  # imgs进行提升维度，从3维扩展到4维

    b, h, w, c = imgs.shape
    patch_h, patch_w = patch_size
    stride_h, stride_w = stride_size

    # padding 填充
    left_h, left_w = (h - patch_h) % stride_h, (w - patch_w) % stride_w
    # 各边需要填充的长短，之所还要取模是因为left_h,left_w可能为0，代表根本不可能需要填充。
    global pad_h
    global pad_w
    pad_h, pad_w = (stride_h - left_h) % stride_h, (stride_w - left_w) % stride_w

    # 在竖直方向上进行填充，填充后，将原图置于中间，顶部和底部使用原图的镜像进行填充。
    if pad_h:
        pad_imgs = np.empty((b, h + pad_h, w, c), dtype=imgs.dtype)
        # 这部分位置用来放置原图
        start_y = pad_h // 2
        end_y = start_y + h

        for i, img in enumerate(imgs):
            # 中间部分设为原图
            pad_imgs[i, start_y:end_y, :, :] = img
            # 上下两编填充部分等于用原图的垂直镜像翻转填充
            pad_imgs[i, :start_y, :, :] = img[:start_y, :, :][::-1]
            pad_imgs[i, end_y:, :, :] = img[h - (pad_h - pad_h // 2):, :, :][::-1]
        imgs = pad_imgs

    if pad_w:
        h = imgs.shape[1]
        pad_imgs = np.empty((b, h, w + pad_w, c), dtype=imgs.dtype)
        # 这部分位置用来放置原图
        start_x = pad_w // 2
        end_x = start_x + w

        for i, img in enumerate(imgs):
            # 中间部分设为原图
            pad_imgs[i, :, start_x:end_x, :] = img
            # 上下两编填充部分等于用原图的垂直镜像翻转填充
            pad_imgs[i, :, :start_x, :] = img[:, :start_x, :][:, ::-1]
            pad_imgs[i, :, end_x:, :] = img[:, w - (pad_w - pad_w // 2):, :][:, ::-1]
        imgs = pad_imgs
    return imgs


# 恢复镜像padding前的样子
def img_padding_recover(imgs, img_size: tuple):
    img_h, img_w = img_size
    b, h, w, c = imgs.shape
    pad_img = np.zeros((b, img_h, img_w, c))
    for i, img in enumerate(imgs):
        global pad_h
        global pad_w
        start_y = pad_h // 2
        end_y = start_y + img_h
        start_x = pad_w // 2
        end_x = start_x + img_w
        img = imgs[i, start_y:end_y, start_x:end_x]
        pad_img[i] = img
    return pad_img


# 图像切片，按序切片
def extract_ordered_patches(imgs, patch_size: tuple, stride_size: tuple):
    """从上到下，从左到右依次对图像进行切片，每个图像块大小相同，相互之间间隔相等
        Args:
            imgs:单张/一个批次的图像，(h,w,c)/(b,h,w,c);
            patch_size: 图像块的大小，（patch_h,patch_w）
            stride_size:每个图像块的间隔，（stride_h,strde_w）

        returns:
            numpy.array:(n_patches,patch_h,patch_w)
    """
    # 判断图片的维数
    assert imgs.ndim > 2

    if imgs.ndim == 3:
        # (h,w,c)->(1,h,w,c)
        imgs = np.expand_dims(imgs, axis=0)  # imgs进行提升维度，从3维扩展到4维

    b, h, w, c = imgs.shape
    patch_h, patch_w = patch_size
    stride_h, stride_w = stride_size

    # 图像各边必须能切出整数数量的图像块
    assert (h - patch_h) % stride_h == 0 and (w - patch_w) % stride_w == 0

    # y方向上的切片数量

    n_patches_y = (h - patch_h) // stride_h + 1

    # x方向上的切片数量
    n_patches_x = (w - patch_w) // stride_w + 1

    # 每张图像的切片数量
    n_patches_per_img = n_patches_y * n_patches_x

    # 切片总数
    # print("切片总数")
    n_patches = n_patches_per_img * b
    # print(n_patches)
    # 设置图像块大小
    patches = np.empty((n_patches, patch_h, patch_w, c), dtype=np.float32)
    patch_idx = 0

    # 依次对每张图像进行切块
    for img in imgs:
        # 从上到下，从左到右的依次切块
        for i in range(n_patches_y):
            for j in range(n_patches_x):
                y1 = i * stride_h
                y2 = y1 + patch_h
                x1 = j * stride_w
                x2 = x1 + patch_w

                patches[patch_idx] = img[y1:y2, x1:x2]
                patch_idx += 1

    return np.array(patches)


# 将切片后的图像恢复成原来的图像
def rebuild_images(patches, img_size: tuple, stride_size: tuple):
    """

    :param patches: (n_patches,patch_h,patch_w,c)
    :param img_size: 图像尺寸,(img_h,img_w)
    :param stride_size: 每个图像块的间隔大小,(stride_h,stride_w_
    :return: images:(b,img_h,img_w,c)
    """
    assert patches.ndim == 4

    img_h, img_w = img_size
    stride_h, stride_w = stride_size
    n_patches, patch_h, patch_w, c = patches.shape
    assert (img_h - patch_h) % stride_h == 0 and (img_w - patch_w) % stride_w == 0
    # yx方向上的切片数量
    n_patches_y = (img_h - patch_h) // stride_h + 1
    n_patches_x = (img_w - patch_w) // stride_w + 1
    # 每幅图像上的总共的切片数量
    n_patches_per_img = n_patches_y * n_patches_x
    batch_size = n_patches // n_patches_per_img

    # 设置图像大小
    # zeros() 创建指定大小的数组，并用以0来填充，zeros(shape,dtype=float,order='C') 分别表示为形状，数据类型，用于C的行数组
    imgs = np.zeros((batch_size, img_h, img_w, c))
    # 图像块之间可能有重叠，因此最后需要除以重复的次数求平均
    # np.zeros_like() 返回给定形状和类型的数组，即给定数组，并带有0
    weights = np.zeros_like(imgs)

    # 并不是直接将切片放入图像的原来位置，因为切片之间可能存在重叠部分，要使用求和
    # enumerate() 枚举，将一个可遍历的数据对象，组合为一个索引序列，同时列出数据和数据下标
    # zip 将对象中对应的元素打包成一个个元组，然后返回这些元组组成的列表，python3 返回一个对象
    for img_idx, (img, weights) in enumerate(zip(imgs, weights)):
        start = img_idx * n_patches_per_img

        for i in range(n_patches_y):
            for j in range(n_patches_x):
                y1 = i * stride_h
                y2 = y1 + patch_h
                x1 = j * stride_w
                x2 = x1 + patch_w
                patch_idx = start + i * n_patches_x + j
                img[y1:y2, x1:x2] += patches[patch_idx]
                weights[y1:y2, x1:x2] += 1

    # 对于重叠部分求均值
    imgs /= weights

    return imgs.astype(patches.dtype)


# 网络的输出是一个 （batch_size, num_classes, h, w) shape 的张量
# label 是一个 （batch_size, 1, h, w) shape 的张量


def mask2one_hot(label, out):
    """
    label: 标签图像 # （batch_size, 1, h, w)
    out: 网络的输出
    """
    num_classes = out.shape[1]  # 分类类别数

    current_label = label.squeeze(1)  # （batch_size, 1, h, w) ---> （batch_size, h, w)

    batch_size, h, w = current_label.shape[0], current_label.shape[1], current_label.shape[2]

    # print(h, w, batch_size)

    one_hots = []
    for i in range(num_classes):
        tmplate = torch.ones(batch_size, h, w)  # （batch_size, h, w)
        tmplate[current_label != i] = 0
        tmplate = tmplate.view(batch_size, 1, h, w)  # （batch_size, h, w) --> （batch_size, 1, h, w)

        one_hots.append(tmplate)

    onehot = torch.cat(one_hots, dim=1)

    return onehot


def display_image_grid(images_filenames, predicted_masks):
    for i, image_filename in enumerate(images_filenames):
        predicted_mask = predicted_masks[i]


def post_process(img, min_size=10):
    '''
    图像后处理过程
    包括开运算和去除过小像素点。
    返回uint16格式numpy二值数组
    '''
    img = img.cpu()
    img = img.numpy().astype(np.bool)
    b, c, w, h = img.shape
    if c == 1:
        for i in range(b):
            img_tmp = img[i, 0, :, :]
            img_tmp = binary_opening(img_tmp)  # 图像开运算
            remove_small_objects(img_tmp, min_size=min_size, in_place=True)
            img_tmp = ~remove_small_objects(~img_tmp, min_size=min_size)
            img[i, 0, :, :] = img_tmp

    return img.astype(np.uint16)


def analysis(x, y):
    '''
    对输入的两个四维张量[B,1,H,W]进行逐图的DSC、PPV、Sensitivity计算
    其中x表示网络输出的预测值
    y表示实际的预想结果mask
    返回为一个batch中DSC、PPV、Sen的平均值及batch大小
    '''
    x = x.type(dtype=torch.uint8)
    y = y.type(dtype=torch.uint8)  # 保证类型为uint8
    DSC = []
    PPV = []
    Sen = []
    if x.shape == y.shape:
        batch = x.shape[0]
        for i in range(batch):  # 按第一个维度分开

            tmp = torch.eq(x[i], y[i])

            tp = int(torch.sum(torch.mul(x[i] == 1, tmp == 1)))  # 真阳性
            fp = int(torch.sum(torch.mul(x[i] == 1, tmp == 0)))  # 假阳性
            fn = int(torch.sum(torch.mul(x[i] == 0, tmp == 0)))  # 假阴性

            try:
                DSC.append(2 * tp / (fp + 2 * tp + fn))
            except:
                DSC.append(0)
            try:
                PPV.append(tp / (tp + fp))
            except:
                PPV.append(0)
            try:
                Sen.append(tp / (tp + fn))
            except:
                Sen.append(0)


    else:
        sys.stderr.write('Analysis input dimension error')

    DSC = sum(DSC) / batch
    PPV = sum(PPV) / batch
    Sen = sum(Sen) / batch
    return DSC, PPV, Sen, batch

# >out = torch.rand(4,3,384,544)	# 网络的输出
# >label = torch.randint(0,3,(4,1,384,544))	# 图像标签
#
# >oh = mask2one_hot(label,out)
# >oh	# shape ---> (4,3,384,544)


# #对mask进行切片
# test_path = './Ti/train/mask'
# img_Tensor = mask_get_imlist(test_path)
# img_Tensor = np.array(img_Tensor)
#
# # img_Tensor = torch.from_numpy(c)
# # 读取转为list和shape后，并没有改变图片本身。
# # print(c.shape)
# testpath = './Ti/mask_data/'
# #
# # print(img_Tensor.shape)
# # patches = []
# # patches = img_padding(img_Tensor, [512, 512], [512, 512])
# # print("镜像padding填充")  # 镜像填充也没有改变图片本身
# # print(patches.shape)
# # patches = extract_ordered_patches(img_Tensor, [512, 512], [512, 512])
# # print("图片切片")
# # print(patches.shape)
# save_tensor_img(img_Tensor, testpath, 'png')


# base_dir = './Ti/mask_data'
# save_dir = './Ti/mask_data_save'
# label_png = './Ti/mask_data_save/0_r.png'
# label_png_1 = './Ti/mask_data/0_r.png'
# transform2img(base_dir, save_dir)
# lbI = np.asarray(Image.open(label_png))
# # ret,mask = cv2.threshold(lbI,127,255,cv2.THRESH_BINARY)
# print(lbI)

# mask=cv2.imread(label_png_1,0)
# print(mask.shape)
# cv2.imwrite('./Ti/test.png',mask)
# ret,mask = cv2.threshold(mask,10,1,cv2.THRESH_BINARY)
# print(mask.shape)
# ret,mask = cv2.threshold(mask,0,255,cv2.THRESH_BINARY)
# cv2.imwrite('./Ti/test1.png',mask)
# mask=torch.randn(1,512,512)
# mask=mask.numpy()
# ret,mask = cv2.threshold(mask,0,255,cv2.THRESH_BINARY)
# cv2.imwrite('./Ti/test.png',mask)
# print(mask)
