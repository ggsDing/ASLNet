import torch
import numpy as np
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F
from models.FCN_8s import FCN_res34
from utils import initialize_weights

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class ED_FCN(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, pretrained=True):
        super(ED_FCN, self).__init__()
        self.FCN = FCN_res34(in_channels, num_classes, pretrained)
                
        self.low = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False),
                                 nn.BatchNorm2d(64), nn.ReLU())
        self.fuse = nn.Sequential( conv3x3(512+64, 64), nn.BatchNorm2d(64), nn.ReLU())        
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)
        
        #self.classifier_aux = nn.Sequential(conv1x1(512, 64), nn.BatchNorm2d(64), nn.ReLU(), conv1x1(64, num_classes))
        initialize_weights(self.low, self.fuse, self.classifier)

    def forward(self, x):
        x_size = x.size()
        
        x = self.FCN.layer0(x)  #1/2
        x = self.FCN.maxpool(x) #1/4
        x1 = self.FCN.layer1(x) #1/4
        x = self.FCN.layer2(x1) #1/8
        x = self.FCN.layer3(x)  #1/8
        x = self.FCN.layer4(x)
        #aux = self.classifier_aux(x)
        
        x1 = self.low(x1)
        x = torch.cat((F.upsample(x, x1.size()[2:], mode='bilinear'), x1), 1)
        fuse = self.fuse(x)
        out = self.classifier(fuse)
        
        return F.upsample(out, x_size[2:], mode='bilinear') #, F.upsample(aux, x_size[2:], mode='bilinear')
