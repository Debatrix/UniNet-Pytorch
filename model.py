from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F


class FeatNet(nn.Module):
    def __init__(self):
        super(FeatNet, self).__init__()
        self.conv1 = nn.Sequential(OrderedDict([
            ('conv1_a', nn.Conv2d(1, 16, kernel_size=(3, 7), stride=1, padding=(1, 3), bias=False)),
            ('tan1_a', nn.Tanh())
        ]))
        self.conv2 = nn.Sequential(OrderedDict([
            ('pool1_a', nn.AvgPool2d(kernel_size=2, stride=2)),
            ('conv2_a', nn.Conv2d(16, 32, kernel_size=(3, 5), stride=1, padding=(1, 2), bias=False)),
            ('tan2_a', nn.Tanh())
        ]))
        self.conv3 = nn.Sequential(OrderedDict([
            ('pool2_a', nn.AvgPool2d(kernel_size=2, stride=2)),
            ('conv3_a', nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)),
            ('tan3_a', nn.Tanh())
        ]))
        self.fuse_a = nn.Conv2d(112, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x2 = F.interpolate(x2, size=(64, 512), mode='bilinear', align_corners=False)
        x3 = F.interpolate(x3, size=(64, 512), mode='bilinear', align_corners=False)
        x4 = torch.cat((x1, x2, x3), dim=1)
        out = self.fuse_a(x4)
        return out


class MaskNet(nn.Module):
    def __init__(self):
        super(MaskNet, self).__init__()
        self.m_conv1 = nn.Sequential(OrderedDict([
            ('m_conv1_a', nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2, bias=True)),
            ('m_relu_a', nn.ReLU(inplace=True)),
        ]))
        self.m_conv2 = nn.Sequential(OrderedDict([
            ('m_pool1_a', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('m_conv2_a', nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=True)),
            ('m_relu2_a', nn.ReLU(inplace=True))
        ]))
        self.m_score2_a = nn.Sequential(
            OrderedDict([('m_score2_a', nn.Conv2d(32, 2, kernel_size=1, stride=1, bias=True))]))
        self.m_conv3 = nn.Sequential(OrderedDict([
            ('m_pool2_a', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('m_conv3_a', nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True)),
            ('m_relu3_a', nn.ReLU(inplace=True))
        ]))
        self.m_score3_a = nn.Sequential(
            OrderedDict([('m_score3_a', nn.Conv2d(64, 2, kernel_size=1, stride=1, bias=True))]))
        self.m_conv4 = nn.Sequential(OrderedDict([
            ('m_pool3_a', nn.MaxPool2d(kernel_size=4, stride=4)),
            ('m_conv4_a', nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)),
            ('m_relu4_a', nn.ReLU(inplace=True))
        ]))
        self.m_score4_a = nn.Sequential(
            OrderedDict([('m_score4_a', nn.Conv2d(128, 2, kernel_size=1, stride=1, bias=True))]))

    def forward(self, x):
        x1 = self.m_conv2(self.m_conv1(x))
        x2 = self.m_conv3(x1)
        x3 = self.m_score4_a(self.m_conv4(x2))
        x34 = self.m_score3_a(x2) + F.interpolate(x3, size=(16, 128), mode='bilinear', align_corners=False)
        x234 = self.m_score2_a(x1) + F.interpolate(x34, size=(32, 256), mode='bilinear', align_corners=False)
        out = F.interpolate(x234, size=(64, 512), mode='bilinear', align_corners=False)
        return out


if __name__ == '__main__':
    a = torch.rand((1, 1, 64, 512))
    f = FeatNet()
    print(f(a).shape)
    m = MaskNet()
    print(m(a).shape)
