import torch
import numpy as np
import torch.nn as nn
from data_process import interpolation_real
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.nn import Conv2d, Linear, BatchNorm1d, BatchNorm2d

from torch.nn.functional import relu, max_pool2d, dropout, dropout2d
from complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear
from complexFunctions import complex_relu, complex_max_pool2d
from scipy.io import loadmat,savemat
# from tensorboardX import SummaryWriter
import os.path
class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4 * growthRate
        # self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        # self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(x))
        out = self.conv2(F.relu(out))
        out = torch.cat((x, out), 1)
        return out


class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        # self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):  # 和正常的resnet不同的是 我们直接输出x-out
        out = self.conv1(F.relu(x))
        out = torch.cat((x, out), 1)
        return out


class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        # self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(x))
        # out = F.avg_pool2d(out, 2)
        return out

class NewNet_Net(nn.Module):
    def __init__(self, growthRate, depth, reduction, nClasses, bottleneck):
        super(NewNet_Net, self).__init__()

        nDenseBlocks = (depth - 2) // 3  # 每个denseblock有32层 cnn，一共3个denseblock ##需要改一下代码？——不用 正好
        if bottleneck:
            nDenseBlocks //= 2  # //为向下取整——代表有一半用来1x1减小维度了的意思吗？

        nChannels = 2 * growthRate
        self.conv1 = Conv2d(1, 32, 3, 1, 2)
        self.conv2 = Conv2d(32, 16, 1, 1, 0)
        self.conv3 = Conv2d(16, 1, 5, 1, 1)

        self.conv1_desnet = Conv2d(2, nChannels, kernel_size=3, padding=1,
                                   bias=False)
        self.singleLayer1 = SingleLayer(nChannels, growthRate)
        nChannels = nChannels + growthRate
        self.singleLayer2 = SingleLayer(nChannels, growthRate)
        nChannels = nChannels + growthRate
        self.singleLayer3 = SingleLayer(nChannels, growthRate)
        nChannels = nChannels + growthRate
        nOutChannels = 2 * growthRate
        self.trans1 = Transition(nChannels, nOutChannels)  # tansition layer——防止block之间的传递过大

        nChannels = nOutChannels
        self.singleLayer4 = SingleLayer(nChannels, growthRate)
        nChannels = nChannels + growthRate
        self.singleLayer5 = SingleLayer(nChannels, growthRate)
        nChannels = nChannels + growthRate
        self.singleLayer6 = SingleLayer(nChannels, growthRate)
        nChannels = nChannels + growthRate
        nOutChannels = 2 * growthRate
        # nOutChannels = int(math.floor(nChannels * reduction))
        self.trans2 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.singleLayer7 = SingleLayer(nChannels, growthRate)
        nChannels = nChannels + growthRate
        self.singleLayer8 = SingleLayer(nChannels, growthRate)
        nChannels = nChannels + growthRate
        self.singleLayer9 = SingleLayer(nChannels, growthRate)
        nChannels = nChannels + growthRate
        nOutChannels = 2 * growthRate
        self.trans3 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.singleLayer10 = SingleLayer(nChannels, growthRate)
        nChannels = nChannels + growthRate
        self.singleLayer11 = SingleLayer(nChannels, growthRate)
        nChannels = nChannels + growthRate
        self.singleLayer12 = SingleLayer(nChannels, growthRate)
        nChannels = nChannels + growthRate
        nOutChannels = 2 * growthRate
        self.trans4 = Transition(nChannels, nOutChannels)

        self.conv4 = Conv2d(2 * growthRate, 2, 3, 1, 1)

        self.bn1 = ComplexBatchNorm2d(nChannels)
        self.fc = ComplexLinear(nChannels, nClasses)
      

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):

  
        x_out1 = self.conv1_desnet(x)
        x_out = self.singleLayer1(x_out1)
        x_out = self.singleLayer2(x_out)
        x_out = self.singleLayer3(x_out)
      
        x_out = self.trans1(x_out)

        x_out2 = x_out1 - x_out
        x_out = self.singleLayer4(x_out2)
        x_out = self.singleLayer5(x_out)
        x_out = self.singleLayer6(x_out)
        x_out = self.trans2(x_out)

    
        x_out3 = x_out2 - x_out
        x_out = self.singleLayer7(x_out3)
        x_out = self.singleLayer8(x_out)
        x_out = self.singleLayer9(x_out)
        x_out = self.trans3(x_out)

        x_out4 = x_out3 - x_out
        x_out = self.singleLayer10(x_out4)
        x_out = self.singleLayer11(x_out)
        x_out = self.singleLayer12(x_out)
        x_out = self.trans4(x_out)

        x_out5 = x_out4 - x_out
        x_out6 = x_out1 - x_out5
        x_out = self.conv4(x_out6)

        return x_out

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
