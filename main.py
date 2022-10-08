# 导入相关的包
import os
import random

import albumentations as A
import cv2
import numpy as np
import torch.nn as nn
import torch.optim
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import utils as vutils
import Res50_unet
import dataSet
import model
import utils

# 相关超参
params = {
    "lr": 0.0001,
    "batch_size": 4,
    "num_workers": 20,
    "epochs": 1,
    "device": "cuda"
}
_exp_name = "sample"


def preprocess_mask(mask):
    mask = mask.astype(np.float32)
    mask[mask == 1.0] = 0.0
    mask[mask == 0.0] = 1.0
    return mask


# def display_image_grid(images_filenames, images_directory, masks_directory, predicted_masks=None):
#     cols = 3 if predicted_masks else 2
#     rows = len(images_filenames)
#     figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 24))
#     for i, image_filename in enumerate(images_filenames):
#         image = cv2.imread(os.path.join(images_directory, image_filename))
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#         mask = cv2.imread(os.path.join(masks_directory, image_filename.replace(".jpg", ".png")), cv2.IMREAD_UNCHANGED, )
#         mask = preprocess_mask(mask)
#         ax[i, 0].imshow(image)
#         ax[i, 1].imshow(mask, interpolation="nearest")
#
#         ax[i, 0].set_title("Image")
#         ax[i, 1].set_title("Ground truth mask")
#
#         ax[i, 0].set_axis_off()
#         ax[i, 1].set_axis_off()
#
#         if predicted_masks:
#             predicted_mask = predicted_masks[i]
#             ax[i, 2].imshow(predicted_mask, interpolation="nearest")
#             ax[i, 2].set_title("Predicted mask")
#             ax[i, 2].set_axis_off()
#         break
#
#     plt.show()




# # 监控评价指标
# class MetricMonitor:
#     def __init__(self, float_precision=2):
#         self.float_precision = float_precision
#         self.reset()
#
#     def reset(self):
#         self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})
#
#     def update(self, metric_name, val):
#         metric = self.metrics[metric_name]
#
#         metric["val"] += val
#         metric["count"] += 1
#         metric["avg"] = metric["val"] / metric["count"]
#
#     def __str__(self):
#         return " | ".join(
#             [
#                 "{metric_name}: {avg:.{float_precision}f}".format(
#                     metric_name=metric_name, avg=metric["avg"], float_precision=self.float_precision
#                 )
#                 for (metric_name, metric) in self.metrics.items()
#             ]
#         )



dataset_directory = "./Ti"
root_directory = os.path.join(dataset_directory)
testpath = './Ti/test_data/'
# 定义数据集路径，并划分训练集和验证集大小
images_directory = os.path.join(root_directory, "train_data")
masks_directory = os.path.join(root_directory, "mask_data")
test_directory = os.path.join(testpath)
print(masks_directory)
print(test_directory)
images_filenames = list(sorted(os.listdir(images_directory)))
test_images_filenames = list(sorted(os.listdir(test_directory)))

random.seed(42)
random.shuffle(images_filenames)

train_images_filenames = images_filenames[:456]
val_images_filenames = images_filenames[456:-1]

print(len(train_images_filenames), len(val_images_filenames), len(test_images_filenames))

# #  Define a function to preprocess a mask
# * The dataset contains pixel-level trimap segmentation. For each image, there is an associated PNG file with a mask. The size of a mask equals to the size of the related image. Each pixel in a mask image can take one of three values: 1, 2, or 3. 1 means that this pixel of an image belongs to the class pet, 2 - to the class background, 3 - to the class border. Since this example demonstrates a task of binary segmentation (that is assigning one of two classes to each pixel), we will preprocess the mask, so it will contain only two uniques values: 0.0 if a pixel is a background and 1.0 if a pixel is a pet or a border.


# 设置数据增强模式
train_tfm = A.Compose([
    A.RandomGamma(gamma_limit=(20, 20), eps=None, always_apply=False, p=1),
    A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
    A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Blur(blur_limit=4, p=1),
    ToTensorV2()
])
val_transform = A.Compose([
    A.Resize(512, 512),
    ToTensorV2()
])
test_transform =  A.Compose([
    A.Resize(512, 512),
    ToTensorV2()
])
# 构建dataset
train_dataset = dataSet.train_mask_dataSet(train_images_filenames, images_directory, masks_directory, transform=train_tfm, )

val_dataset = dataSet.train_mask_dataSet(val_images_filenames, images_directory, masks_directory, transform=val_transform, )

test_dataset = dataSet.TiTestDataset(test_images_filenames, test_directory, transform=test_transform, )

# 构建loader
train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True,
                          num_workers=params["num_workers"], pin_memory=True, )
val_loader = DataLoader(val_dataset, batch_size=params["batch_size"], shuffle=False, num_workers=params["num_workers"],
                        pin_memory=True, )
test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=False,
                         num_workers=0, pin_memory=True, )

# 创建模型
Model = model(BN_enable=True, resnet_pretrain=False).to(params["device"])
# 早停法，能够忍受多少个epoch内没有improvement
patience = 32
# 确定损失函数，才有crossEntropyLoss
criterion = nn.BCEWithLoss().to(params["device"])
# 确定优化器
optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=1e-5)
# 确定最好的iou，保存
stale = 0
best_iou = 0

for epoch in range(1, params["epochs"] + 1):
    # metric_monitor = MetricMonitor()

    # ---------- Training ----------
    # Make sure the model is in train mode before training.
    train_loss = []
    train_ious=[]
    model.train()

    for batch in tqdm(train_loader, leave=False):
        images, target = batch

        images = images.to(params["device"], non_blocking=True)

        output = model(images)
        # target = utils.mask2one_hot(target, output)
        target = target.to(params["device"], non_blocking=True)
        loss = criterion(output, target)
        # metric_monitor.update("Loss", loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        # train_ious.append(miou)
    train_loss = sum(train_loss) / len(train_loss)
    # Print the information.
    # print(f"[ Train | {epoch + 1:03d}/{params['epochs']+1:03d} ] loss = {train_loss:.5f}, miou = {train_ious:.5f}")
    print(f"[ Train | {epoch :03d}/{params['epochs'] :03d} ] loss = {train_loss:.5f}")
    # ---------- Validation ----------
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    model.eval()
    valid_loss = []
    DSC_sum = 0
    PPV_sum = 0
    Sen_sum = 0

    for batch in tqdm(val_loader, leave=False):
        images, target = batch
        DSC = 0
        PPV = 0
        Sen = 0
        batch_num = 0
        with torch.no_grad():
            output = model(images.to(params["device"]))
            output = torch.ge(output, 0.5).type(dtype=torch.float32)  # 二值化
            target = target.to(params["device"], non_blocking=True)
            loss = criterion(output, target)

            valid_loss.append(loss.item())
        # valid_ious.append(miou)
    valid_loss = sum(valid_loss) / len(valid_loss)

    # Print the information.
    # print(f"[ valid | {epoch + 1:03d}/{params['epochs'] + 1:03d} ] loss = {valid_loss:.5f}, miou = {valid_ious:.5f}")
    print(f"[ valid | {epoch :03d}/{params['epochs']:03d} ] loss = {valid_loss:.5f}")
    #
    # update logs
    # if valid_ious > best_iou:
    #     with open(f"./{_exp_name}_log.txt", "a"):
    #         print(f"[ Valid | {epoch + 1:03d}/{params['epochs']+1:03d} ] loss = {valid_loss:.5f}, acc = {valid_ious:.5f} -> best")
    # else:
    #     with open(f"./{_exp_name}_log.txt", "a"):
    #         print(f"[ Valid | {epoch + 1:03d}/{params['epochs']+1:03d} ] loss = {valid_loss:.5f}, acc = {valid_ious:.5f}")
    #
    # # save models
    # if valid_ious > best_iou:
    #     print(f"Best model found at epoch {epoch}, saving model")
    #     torch.save(model.state_dict(), f"{_exp_name}_best.ckpt")  # only save best to prevent output memory exceed error
    #     best_acc = valid_ious
    #     stale = 0
    # else:
    #     stale += 1
    #     if stale > patience:
    #         print(f"No improvment {patience} consecutive epochs, early stopping")
    #         break

# predictions = predict(model, params, test_dataset, batch_size=4)
# predicted_masks = []
# for predicted_256x256_mask, original_height, original_width in predictions:
#     full_sized_mask = F.resize(
#         predicted_256x256_mask, height=original_height, width=original_width, interpolation=cv2.INTER_NEAREST
#     )
#     predicted_masks.append(full_sized_mask)
#
# example_image_filename = test_images_filenames[0]
# image = cv2.imread(os.path.join(test_directory, example_image_filename))
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# # mask = cv2.imread(os.path.join(masks_directory, example_image_filename.replace(".jpg", ".png")), cv2.IMREAD_UNCHANGED,)
# mask = predicted_masks[0]
# figure, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 10))
# ax.ravel()[0].imshow(image)
# ax.ravel()[0].set_title("Original image")
# ax.ravel()[1].imshow(mask, interpolation="nearest")

# 预测图像
model.eval()
predictions=[]
with torch.no_grad():
    for images in test_loader:

        output = images.to(params["device"], non_blocking=True)
        output=model(output)
        print(output)
        predicted_masks = torch.ge(output, 0.5).type(dtype=torch.float32)  # 二值化
        print(predicted_masks.shape)
        print(predicted_masks)
        # predicted_masks = output.argmax(dim=1)
        predicted_masks = predicted_masks.cpu().numpy()
        for predicted_mask in predicted_masks:
            predictions.append(predicted_mask)

for i, test_images_filename in enumerate(test_images_filenames):
    print(test_images_filename)
    predicted_mask = predictions[i]
    print(predicted_mask)
    # predicted_mask = predicted_mask.permute(1, 2, 0)
    vutils.save_image(output[i, :, :, :], './Ti/res/'+test_images_filename, padding=0)
    # cv2.imwrite('./Ti/res/'+test_images_filename,predicted_mask)
