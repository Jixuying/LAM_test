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
DNCNN_HIDDENS = 18
class DNCNN_Net(nn.Module):

    def __init__(self, BN=True, Dropout=False):
        # 这里 ComplexNet继承父类nn.Module中的init
        super(DNCNN_Net, self).__init__()  # https://www.runoob.com/python/python-func-super.html
        self.dobn = BN
        self.dodrop = Dropout  # 根据传入网络中的参数来决定是否执行dropout或者batch normalization
        self.hidden = []
        self.bns = []
        self.drops = []
        self.conv1 = Conv2d(1, 64, 3, 1, 1)
        self.conv2 = Conv2d(64, 64, 3, 1, 1)
        self.conv3 = Conv2d(64, 1, 3, 1, 1)
        for i in range(DNCNN_HIDDENS):
            conv = Conv2d(64, 64, 3, 1, 1)
            setattr(self, 'conv2_hideen%i' % i, conv)
            self.hidden.append(conv)
            if self.dobn:
                bn = BatchNorm2d(64)
                setattr(self, 'bn%i' % i, bn)
                self.bns.append(bn)

    def forward(self, x):  # forward函数定义了网络的前向传播的顺序
        # outputs = []
        x_out = self.conv1(x)
        # outputs.append(x_out.data)
        x_out = relu(x_out)

        for i in range(DNCNN_HIDDENS):
            x_out = self.hidden[i](x_out)
            # outputs.append(x_out.data)
            if self.dobn:
                x_out = self.bns[i](x_out)

            relu(x_out)

        x_out = self.conv3(x_out)
        # outputs.append(x_out.data)
        # x_out = relu(x_out)
        x_diff = x - x_out
        return x_diff

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
