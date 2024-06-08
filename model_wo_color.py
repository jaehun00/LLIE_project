import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage, transforms
from loss import LossFunction
from PIL import Image
import math

import cv2

class resblock(nn.Module):
    
    def __init__(self, channel):
        super(resblock, self).__init__()
        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channel),
        )
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        skip_connection = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + skip_connection
        x = self.relu(x)
        return x
    
class myBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(myBlock, self).__init__()
        self.res_l1 = resblock(out_c)
        self.res_l2 = resblock(out_c)
        self.res_l3 = resblock(out_c)
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=out_c, out_channels=in_c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_c),
            nn.ReLU()
        )
        
    def forward(self, x):

        x = self.in_conv(x)
        x_down2 = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        x_down4 = F.interpolate(x_down2, scale_factor=0.5, mode='bilinear')
        
        x_reup2 = F.interpolate(x_down4, scale_factor=2, mode='bilinear')
        x_reup = F.interpolate(x_reup2, scale_factor=2, mode='bilinear')
        
        Laplace_1 = x - x_reup
        Laplace_2 = x_down2 - x_reup2
        
        scale1 = self.res_l1(x_down4)
        scale2 = self.res_l2(Laplace_2)
        scale3 = self.res_l3(Laplace_1)
        
        output1 = scale1
        output2 = F.interpolate(output1, scale_factor=2, mode='bilinear') + scale2
        output3 = F.interpolate(output2, scale_factor=2, mode='bilinear') + scale3
        
        output3 = self.out_conv(output3)

        return output3
    
class EnhanceNetwork(nn.Module):
    def __init__(self, layers, channels):
        super(EnhanceNetwork, self).__init__()

        self.myblock = myBlock(3, 16)
        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.conv)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        fea = self.in_conv(input)
        for conv in self.blocks:
            fea = fea + conv(fea)
        
        fea = fea + self.myblock(fea)

        fea = self.out_conv(fea)

        illu = fea + input
        illu = torch.clamp(illu, 0.0001, 1)

        return illu
    
class CalibrateNetwork(nn.Module):
    def __init__(self, layers, channels):
        super(CalibrateNetwork, self).__init__()
        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation

        self.myblock = myBlock(16, 16)
        self.layers = layers

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.convs)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        fea = self.in_conv(input)
        for conv in self.blocks:
            fea = fea + conv(fea)

        fea = fea + self.myblock(fea)

        fea = self.out_conv(fea)

        delta = input - fea
        
        return delta
        
class Network(nn.Module):

    def __init__(self, stage=3):
        super(Network, self).__init__()
        self.stage = stage
        self.enhance = EnhanceNetwork(layers=1, channels=3)
        self.calibrate = CalibrateNetwork(layers=2, channels=16)
        self._criterion = LossFunction()

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def forward(self, input):

        ilist, rlist, inlist, attlist = [], [], [], []
        input_op = input

        for i in range(self.stage):
            inlist.append(input_op)

            i = self.enhance(input_op)
            r = input/i
            r = torch.clamp(r, 0, 1)
            att = self.calibrate(r)

            input_op = att + input

            ilist.append(i)
            rlist.append(r)
            attlist.append(torch.abs(att))
            
        return ilist, rlist, inlist, attlist

    def _loss(self, input):
        i_list, en_list, in_list, _= self(input)
        losses = 0
        
        for i in range(self.stage):
            loss = self._criterion(in_list[i], i_list[i])
            losses += loss

        return losses

class Finetunemodel(nn.Module):

    def __init__(self, weights):
        super(Finetunemodel, self).__init__()
        self.enhance = EnhanceNetwork(layers=1, channels=3)
        self._criterion = LossFunction()

        base_weights = torch.load(weights)
        pretrained_dict = base_weights
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def forward(self, input):
        i = self.enhance(input)
        r = input / i
        r = torch.clamp(r, 0, 1)
        return i, r

    def _loss(self, input):
        i, r = self(input)
        loss = self._criterion(input, i)
        return loss