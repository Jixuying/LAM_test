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

class SRCNN_Net(nn.Module):

    def __init__(self):
        # 这里 ComplexNet继承父类nn.Module中的init
        super(SRCNN_Net, self).__init__()  # https://www.runoob.com/python/python-func-super.html
        self.conv1 = Conv2d(2, 64, 9, 1, 4)
        self.conv2 = Conv2d(64, 32, 1, 1, 0)
        self.conv3 = Conv2d(32, 2, 5, 1, 2)

    def forward(self, x):  # forward函数定义了网络的前向传播的顺序
        # outputs = []
        x = self.conv1(x)
        # outputs.append(x)
        x = relu(x)
        x = self.conv2(x)
        # outputs.append(x)
        x = relu(x)
        x = self.conv3(x)
        # outputs.append(x)
        # x = relu(x)
        return x

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
