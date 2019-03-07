import os
import sys
import cv2
import torch
from torchvision.models import vgg
import Library.Dataset
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#import ipdb


# 角度偏差损失
def OrientationLoss(orient, angleDiff, confGT):
    #
    # orid = [sin(delta), cos(delta)] shape = [batch, bins, 2]
    # angleDiff = GT - center, shape = [batch, bins]
    #
    [batch, _, bins] = orient.size()
    cos_diff = torch.cos(angleDiff)
    sin_diff = torch.sin(angleDiff)
    cos_ori = orient[:, :, 0]
    sin_ori = orient[:, :, 1]
    mask1 = (confGT != 0)
    mask2 = (confGT == 0)
    count = torch.sum(mask1, dim=1).type(torch.FloatTensor).cuda()
    tmp = cos_diff * cos_ori + sin_diff * sin_ori  # cos(diff - ori) = cos(diff) * cos(ori) + sin(diff) * sin(ori)
    tmp[mask2] = 0
    # total = torch.sum(tmp, dim = 1)
    # count = count.type(torch.FloatTensor).cuda()
    # total = total / count
    # return -torch.sum(total) / batch
    total = torch.sum(2 - 2 * torch.mean(tmp, dim=0)) / count

    return torch.mean(total)


# TODO: 试试Leaky ReLU
class Model(nn.Module):
    def __init__(self, features=None, bins=2, w = 0.4, mode='training'):
        super(Model, self).__init__()
        self.bins = bins
        self.w = w
        self.features = features  # backbone网络
        self.mode = mode
        self.orientation = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, bins*2) # to get sin and cos
                )
        self.confidence = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, bins)
                    #nn.Softmax()
                    #nn.Sigmoid()
                )
        self.dimension = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 512),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(512, 512),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(512, 3)
                )
        self.viewpoint = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 512),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(512, 512),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(512, 16)
                )

    def forward(self, x):
        x = self.features(x)  # 512 x 7 x 7
        x = x.view(-1, 512 * 7 * 7)
        # 三个分支输出
        # 输出角度偏差
        orientation = self.orientation(x)
        orientation = orientation.view(-1, self.bins, 2)
        orientation = F.normalize(orientation, dim=2)  # 归一化实现sin与cos的约束
        # 输出角度bin的置信度
        confidence = self.confidence(x)
        if self.mode != 'training':
            confidence = nn.functional.softmax(confidence)
        # 输出三维尺寸
        dimension = self.dimension(x)
        # 输出视角分类
        viewpoint= self.viewpoint(x)
        return orientation, confidence, dimension, viewpoint
