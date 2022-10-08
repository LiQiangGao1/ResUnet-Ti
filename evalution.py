import torch
import sys
import sys

import torch


def average(list):
    s = 0
    for item in list:
        s += item
    return s / len(list)


def sum(list):
    s = 0
    for item in list:
        s += item
    return s


def analysis(x, y):
    """
    对输入的两个四维张量[B,1,H,W]进行逐图的DSC、PPV、Sensitivity计算
    其中x表示网络输出的预测值
    y表示实际的预想结果mask
    返回为一个batch中DSC、PPV、Sen的平均值及batch大小
    """
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


