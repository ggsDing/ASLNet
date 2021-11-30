import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class FCDiscriminator(nn.Module):
    def __init__(self, in_channels, ndf = 16):
        super(FCDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        #self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
        #self.conv5 = nn.Conv2d(ndf*8, ndf*8, kernel_size=4, stride=2, padding=1)
        #self.conv6 = nn.Conv2d(ndf*8, ndf*8, kernel_size=4, stride=2, padding=1)        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        self.classifier = nn.Conv2d(ndf*4, 1, kernel_size=1)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.interpolate(x, scale_factor=0.5, mode='area') #1/2
        x = self.conv1(x) #1/4
        x = self.leaky_relu(x)
        x1 = self.conv2(x) #1/8
        x = self.leaky_relu(x1)
        x = self.conv3(x) #1/16
        x = self.leaky_relu(x)
        #x = self.conv4(x) #1/32
        #x = self.leaky_relu(x)
        #x = self.conv5(x) #1/64
        #x = self.leaky_relu(x)
        
        out = self.classifier(x)
        #out = self.sigmoid(out)

        return out
        
class MSDiscriminator(nn.Module):
    def __init__(self, in_channels, ndf = 16):
        super(MSDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)       
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)        
        self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=1)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.interpolate(x, scale_factor=0.5, mode='area') #1/2
        x1 = self.conv1(x) #1/4
        x1 = self.leaky_relu(x1)
        x2 = self.conv2(x1) #1/8
        x2 = self.leaky_relu(x2)
        x3 = self.conv3(x2) #1/16
        x3 = self.leaky_relu(x3)
        x4 = self.conv4(x3) #1/32
        x4 = self.leaky_relu(x4)
        
        x1 = F.interpolate(x1, x4.size()[2:], mode='bilinear')
        x2 = F.interpolate(x2, x4.size()[2:], mode='bilinear')
        x3 = F.interpolate(x3, x4.size()[2:], mode='bilinear')
        out = self.classifier(x4)

        return [x1, x2, x3, x4], out

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),  #1/4
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),  #1/8, 256
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1), #1/16, 128
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, stride=1, padding=1), #1/16, 64
                    nn.InstanceNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=0.5, mode='area') #1/2
        x =  self.model(x)
        # Average pooling and flatten
        return x #F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)