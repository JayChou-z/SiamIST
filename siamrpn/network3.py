import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from .transforms import ToTensor

from torchvision.models import alexnet
from torch.autograd import Variable
from torch import nn

from IPython import embed
from .config import config

import numpy as np
import torch
from torch import nn
from torch.nn import init

from torch.autograd import Function


class SRMLayer(nn.Module):
    def __init__(self, channel, reduction=None):
        # Reduction for compatibility with layer_block interface
        super(SRMLayer, self).__init__()

        # CFC: channel-wise fully connected layer
        self.cfc = nn.Conv1d(channel, channel, kernel_size=2, bias=False,
                             groups=channel)
        self.bn = nn.BatchNorm1d(channel)

    def forward(self, x):
        b, c, _, _ = x.size()

        # Style pooling
        # AvgPool（全局平均池化）：
        mean = x.view(b, c, -1).mean(-1).unsqueeze(-1)
        # StdPool（全局标准池化）
        std = x.view(b, c, -1).std(-1).unsqueeze(-1)
        u = torch.cat((mean, std), -1)  # (b, c, 2)

        # Style integration
        # CFC（全连接层）
        z = self.cfc(u)  # (b, c, 1)
        # BN（归一化）
        z = self.bn(z)
        # Sigmoid
        g = torch.sigmoid(z)

        g = g.view(b, c, 1, 1)
        return x * g.expand_as(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class SiamRPNNet(nn.Module):
    def __init__(self, ):
        super(SiamRPNNet, self).__init__()

        self.featureExtract = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=2),  # 0） stride=2
            nn.BatchNorm2d(96),  # 1）
            nn.MaxPool2d(3, stride=2),  # 2） stride=2
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 256, 5),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3, stride=2),  # 6） stride=2
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 384, 3),
            nn.BatchNorm2d(384),  # 9
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3),
            nn.BatchNorm2d(256),  # 15
        )
        self.srm = SRMLayer(256)

        self.ca = ChannelAttention(256)
        self.sa = SpatialAttention()

        self.anchor_num = config.anchor_num  # 每一个位置有5个anchor 5
        self.input_size = config.instance_size  # 271  检测帧大小
        self.score_displacement = int((self.input_size - config.exemplar_size) / config.total_stride)  # （271-127）/8=18
        self.conv_cls1 = nn.Conv2d(256, 256 * 2 * self.anchor_num, kernel_size=3, stride=1, padding=0)
        self.conv_r1 = nn.Conv2d(256, 256 * 4 * self.anchor_num, kernel_size=3, stride=1, padding=0)

        self.conv_cls2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.conv_r2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.regress_adjust = nn.Conv2d(4 * self.anchor_num, 4 * self.anchor_num, 1)  # 1x1的卷积？

    def forward(self, template, detection):
        N = template.size(0)  # N=16

        template_feature = self.featureExtract(template)  # [32,256,6,6]

        template_feature1 = self.srm(template_feature) * template_feature  #
        template_feature2 = self.ca(template_feature) * template_feature
        template_feature3 = self.sa(template_feature2) * template_feature2
        template_feature = template_feature3 + template_feature1

        detection_feature = self.featureExtract(detection)  # [32,256,24,24]

        detection_feature1 = self.srm(detection_feature) * detection_feature
        detection_feature2 = self.ca(detection_feature) * detection_feature
        detection_feature3 = self.sa(detection_feature2) * detection_feature2
        detection_feature = detection_feature3 + detection_feature1

        kernel_score = self.conv_cls1(template_feature).view(N, 2 * self.anchor_num, 256, 4,
                                                             4)  # 32,2*5,256,4,4  view 重新调整形状

        kernel_regression = self.conv_r1(template_feature).view(N, 4 * self.anchor_num, 256, 4, 4)  # 32,4*5,256,4,4

        conv_score = self.conv_cls2(detection_feature)  # 32,256,22,22#对齐操作
        conv_regression = self.conv_r2(detection_feature)  # 32,256,22,22
        ##组卷积 类别分支 互相关操作
        conv_scores = conv_score.reshape(1, -1, self.score_displacement + 4,
                                         self.score_displacement + 4)  # 1,32x256,22,22
        # 1，8192,22,22
        score_filters = kernel_score.reshape(-1, 256, 4, 4)  # 32x10,256,4,4

        pred_score = F.conv2d(conv_scores, score_filters, groups=N).reshape(N, 10, self.score_displacement + 1,
                                                                            self.score_displacement + 1)  # F.CONV2D卷积操作
        # 32,10,19,19
        conv_reg = conv_regression.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)
        # 32,256,22,22
        reg_filters = kernel_regression.reshape(-1, 256, 4, 4)
        # 640,256, 4, 4
        pred_regression = self.regress_adjust(
            F.conv2d(conv_reg, reg_filters, groups=N).reshape(N, 20, self.score_displacement + 1,
                                                              self.score_displacement + 1))
        # 32, 20, 19, 19
        return pred_score, pred_regression

    def track_init(self, template):
        N = template.size(0)  # 1
        template_feature = self.featureExtract(template)  # 输出 [1, 256, 6, 6]
        template_feature1 = self.srm(template_feature) * template_feature
        template_feature2 = self.ca(template_feature) * template_feature
        template_feature3 = self.sa(template_feature2) * template_feature2
        template_feature = template_feature3 + template_feature1

        # kernel_score=1,2x5,256,4,4   kernel_regression=1,4x5, 256,4,4
        kernel_score = self.conv_cls1(template_feature).view(N, 2 * self.anchor_num, 256, 4,
                                                             4)  # 1,256,6,6  1,256*2*5,4,4  1,2*5,256,4,4
        kernel_regression = self.conv_r1(template_feature).view(N, 4 * self.anchor_num, 256, 4, 4)  # 1 20 256 4 4
        self.score_filters = kernel_score.reshape(-1, 256, 4, 4)  # 2x5, 256, 4, 4
        self.reg_filters = kernel_regression.reshape(-1, 256, 4, 4)  # 4x5, 256, 4, 4      得到模板构成的核

    def track(self, detection):
        N = detection.size(0)
        detection_feature = self.featureExtract(detection)  # N,256,24,24

        detection_feature1 = self.srm(detection_feature) * detection_feature
        detection_feature2 = self.ca(detection_feature) * detection_feature
        detection_feature3 = self.sa(detection_feature2) * detection_feature2
        detection_feature = detection_feature3 + detection_feature1

        conv_score = self.conv_cls2(detection_feature)  # 输入通道256，输出通道256，kernel=3，stride=1，padding[1,256,22,22]
        conv_regression = self.conv_r2(detection_feature)  # 输入通道256，输出通道256，kernel=3，stride=1，padding[1,256,22,22]

        conv_scores = conv_score.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)  # 1，256,22,22
        pred_score = F.conv2d(conv_scores, self.score_filters, groups=N).reshape(N, 10, self.score_displacement + 1,
                                                                                 self.score_displacement + 1)  # 32,1,256,22,22  +  32 ,2*5,256,4,4 =>32,10,19,19 =>32 10,19,19
        conv_reg = conv_regression.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)
        pred_regression = self.regress_adjust(
            F.conv2d(conv_reg, self.reg_filters, groups=N).reshape(N, 20, self.score_displacement + 1,
                                                                   self.score_displacement + 1))  # 32,1,256,22,22 + 32,4*5,256,4,4 =>32,20,19,19 =>32,20,19,19
        return pred_score, pred_regression  # 特征图
