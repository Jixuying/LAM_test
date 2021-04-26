import torch  NewNet_ComplexNet
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
        self.conv1 = ComplexConv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        # self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = ComplexConv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, xr, xi):
        xr_out, xi_out = self.conv1(complex_relu(xr, xi))
        xr_out, xi_out = self.conv2(complex_relu(xr_out, xi_out))
        xr_out = torch.cat((xr, xr_out), 1) #需要改一下 但是没有用到这个class，所以先不管
        xi_out = torch.cat((xi, xi_out), 1)
        return xr_out, xi_out


class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        # self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = ComplexConv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, xr, xi):  # 和正常的resnet不同的是 我们直接输出x-out
        xr_out, xi_out = complex_relu(xr, xi)
        xr_out, xi_out = self.conv1(xr_out, xi_out)
        xr_out = torch.cat((xr, xr_out), 1)  # 需要改一下 但是没有用到这个class，所以先不管
        xi_out = torch.cat((xi, xi_out), 1)
        return xr_out, xi_out


class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        # self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = ComplexConv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, xr, xi):
        xr_out, xi_out = complex_relu(xr, xi)
        xr_out, xi_out = self.conv1(xr_out, xi_out)
        # out = F.avg_pool2d(out, 2)
        return xr_out, xi_out

class NewNet_ComplexNet(nn.Module):
    def __init__(self, growthRate, depth, reduction, nClasses, bottleneck):
        super(NewNet_ComplexNet, self).__init__()

        nDenseBlocks = (depth - 2) // 3  # 每个denseblock有32层 cnn，一共3个denseblock ##需要改一下代码？——不用 正好
        if bottleneck:
            nDenseBlocks //= 2  # //为向下取整——代表有一半用来1x1减小维度了的意思吗？

        nChannels = 2 * growthRate
        self.conv1 = ComplexConv2d(1, 32, 3, 1, 2)
        self.conv2 = ComplexConv2d(32, 16, 1, 1, 0)
        self.conv3 = ComplexConv2d(16, 1, 5, 1, 1)



        self.conv1_desnet = ComplexConv2d(1, nChannels, kernel_size=3, padding=1,
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

        self.conv4 = ComplexConv2d(2 * growthRate, 1, 3, 1, 1)

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
        with torch.no_grad():
            xr = x[:, 0, :, :]  # x就是传进来的data
            # imaginary part to zero
            # xi = torch.zeros(xr.shape, dtype=xr.dtype, device=xr.device)
            xi = x[:, 1, :, :]

            xr = xr[:, None, :, :]
            xi = xi[:, None, :, :]
     
        xr_out1, xi_out1 = self.conv1_desnet(xr, xi)
        xr_out, xi_out = self.singleLayer1(xr_out1, xi_out1)
        xr_out, xi_out = self.singleLayer2(xr_out, xi_out)
        xr_out, xi_out = self.singleLayer3(xr_out, xi_out)
        # xr_out, xi_out = self.singleLayer4(xr_out, xi_out)
        xr_out, xi_out = self.trans1(xr_out, xi_out)

        xr_out2 = xr_out1 - xr_out
        xi_out2 = xi_out1 - xi_out
        xr_out, xi_out = self.singleLayer4(xr_out2, xi_out2)
        xr_out, xi_out = self.singleLayer5(xr_out, xi_out)
        xr_out, xi_out = self.singleLayer6(xr_out, xi_out)
        # xr_out, xi_out = self.singleLayer4(xr_out, xi_out)
        xr_out, xi_out = self.trans2(xr_out, xi_out)

        # xr_out, xi_out = self.trans2(self.dense2(xr_out2, xi_out2))
        xr_out3 = xr_out2 - xr_out
        xi_out3 = xi_out2 - xi_out
        xr_out, xi_out = self.singleLayer7(xr_out3, xi_out3)
        xr_out, xi_out = self.singleLayer8(xr_out, xi_out)
        xr_out, xi_out = self.singleLayer9(xr_out, xi_out)
        # xr_out, xi_out = self.singleLayer4(xr_out, xi_out)
        xr_out, xi_out = self.trans3(xr_out, xi_out)

        # xr_out, xi_out = self.trans3(self.dense3(xr_out3, xi_out3))
        xr_out4 = xr_out3 - xr_out
        xi_out4 = xi_out3 - xi_out
        xr_out, xi_out = self.singleLayer10(xr_out4, xi_out4)
        xr_out, xi_out = self.singleLayer11(xr_out, xi_out)
        xr_out, xi_out = self.singleLayer12(xr_out, xi_out)
        # xr_out, xi_out = self.singleLayer4(xr_out, xi_out)
        xr_out, xi_out = self.trans4(xr_out, xi_out)
        # out = self.trans3(self.dense3(xr_out4, xi_out4))

        xr_out5 = xr_out4 - xr_out
        xi_out5 = xi_out4 - xi_out
        xr_out6 = xr_out1 - xr_out5
        xi_out6 = xi_out1 - xi_out5
        xr_out, xi_out = self.conv4(xr_out6, xi_out6)
        x_final_out = torch.zeros(len(xr[:, 0, 0, 0]), 2, len(xr[1, 0, :, 1]), len(xr[1, 0, 1, :]))
        x_final_out[:, 0, :, :] = xr_out[:, 0, :, :]
        x_final_out[:, 1, :, :] = xi_out[:, 0, :, :]
        # out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))
        # out = F.log_softmax(self.fc(out))
        return x_final_out

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
