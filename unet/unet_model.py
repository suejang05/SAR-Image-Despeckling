import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms,datasets
import torch.nn.functional as F
from .cbam import CBAM

class UNet(nn.Module):
    def __init__(self, n_channels=1, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        
        # CBR = Conv + Batch Norm + LeakyRelu
        def CBR(input_channel, output_channel, kernel_size=3, stride=1, padding=1):
            layer = nn.Sequential (
                nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(num_features=output_channel),
                nn.LeakyReLU(),
                CBAM(output_channel)
            )
            return layer
        
        # Encoding
        self.conv1 = nn.Sequential( #channel 수를 2배 늘림
            CBR(1, 8, 3, 1),
            CBR(8, 8, 3, 1)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # image size를 2배 줄임
        
        self.conv2 = nn.Sequential(
            CBR(8, 16, 3, 1),
            CBR(16, 16, 3, 1)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Sequential(
            CBR(16, 32, 3, 1),
            CBR(32, 32, 3, 1)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) 
        
        self.conv4 = nn.Sequential(
            CBR(32, 64, 3, 1),
            CBR(64, 64, 3, 1),
            nn.Dropout (p=0.5)
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bottleNeck = nn.Sequential(
            CBR(64, 128, 3, 1),
            CBR(128, 128, 3, 1)
        )
        
        
        # Decoding
        self.upconv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2) #image size를 2배 늘림
        self.ex_conv1 = nn.Sequential( # Channel 수를 2배 줄임
            CBR(128, 64, 3, 1), 
            CBR(64, 64, 3, 1)
        )
        
        self.upconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.ex_conv2 = nn.Sequential(
            CBR(64, 32, 3, 1),
            CBR(32, 32, 3, 1)
        )
        
        self.upconv3 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2)
        self.ex_conv3 = nn.Sequential(
            CBR(32, 16, 3, 1),
            CBR(16, 16, 3, 1)
        )
        
        self.upconv4 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=2, stride=2)
        self.ex_conv4 = nn.Sequential(
            CBR(16, 8, 3, 1),
            CBR(8, 8, 3, 1)
        )
        
        self.fc = nn.Conv2d(8, 1, 1, 1)
        
    def forward(self, x):
        skips = [x]
        
        layer1 = self.conv1(x)
        out = self.pool1(layer1)
        
        layer2 = self.conv2(out)
        out = self.pool2(layer2)
        
        layer3 = self.conv3(out)
        out = self.pool3(layer3)
        
        layer4 = self.conv4(out)
        out = self.pool4(layer4)
        
        bottleNeck = self.bottleNeck(out)
        
        upconv1 = self.upconv1(bottleNeck)
        cat1 = torch.cat((transforms.CenterCrop((upconv1.shape[2], upconv1.shape[3]))(layer4), upconv1), dim=1)
        ex_layer1 = self.ex_conv1(cat1)
        
        upconv2 = self.upconv2(ex_layer1)
        cat2 = torch.cat((transforms.CenterCrop((upconv2.shape[2], upconv2.shape[3]))(layer3), upconv2), dim=1)
        ex_layer2 = self.ex_conv2(cat2)
        
        upconv3 = self.upconv3(ex_layer2)
        cat3 = torch.cat((transforms.CenterCrop((upconv3.shape[2], upconv3.shape[3]))(layer2), upconv3), dim=1)
        ex_layer3 = self.ex_conv3(cat3)
        
        upconv4 = self.upconv4(ex_layer3)
        cat4 = torch.cat((transforms.CenterCrop((upconv4.shape[2], upconv4.shape[3]))(layer1), upconv4), dim=1)
        out = self.ex_conv4(cat4)

        out = self.fc(out)
        
        return out
