"""Utilisation : 

    model = UNet(in_channels, out_channels, netType="default", convType="conv2d", doubleLayer=False)

- **in_channels** : Nombre de channel en entrée
- **out_channels** : Nombre de channel de sortie (en théorie 3)
- **netType** : "partial" pour que les convolutions soient partielles, sinon classique
- **convType** : "conv2d" pour convolution classique, sinon gated conv
- **doubleLayer** : Si `True` le réseau sera plus profond, il apprendra plus lentement mais mieux en théorie


---
"""

__all__ = []

import torch
import torch.nn.functional as F
from torch import nn

import impaintingLib.model.layer as layer

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, netType, convType, activation='relu'):
        super().__init__()
        self.netType  = netType
        self.convType = convType

        if "conv2d" in self.convType :
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        else : 
            # normalement dilatation=2 mais erreur ici
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, dilation=1)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, dilation=1)

        self.convPartial1 = layer.PartialConv2d(in_channels, out_channels, 3, padding=1, return_mask=True)
        self.convPartial2 = layer.PartialConv2d(out_channels, out_channels, 3, padding=1, return_mask=True)

        if activation == 'leakyrelu':
            self.activtion = nn.LeakyReLU(0.2)
        else: 
            self.activtion = nn.ReLU()

    def forward(self, x, m):

        if "partial" in self.netType :
            x, m = self.convPartial1(x, m)
            x = self.activtion(x)
            x, m = self.convPartial2(x, m)
            x = self.activtion(x)

        else : 
            x = self.conv1(x)
            x = self.activtion(x)
            x = self.conv2(x)
            x = self.activtion(x)
        return x, m


class DownSampleBlock(nn.Module):

    def __init__(self, in_channels, out_channels, netType, convType):
        super().__init__()
        self.netType = netType

        self.conv_block = DoubleConv(in_channels, out_channels, netType, convType)
        self.maxpool = nn.MaxPool2d(2) 

    def forward(self, x, m):

        x_skip, m_skip = self.conv_block(x, m)
        x = self.maxpool(x_skip)

        if "partial" in self.netType :
            m = self.maxpool(m)
        return x, m , x_skip, m_skip

class UpSampleBlock(nn.Module):

    def __init__(self, in_channels, out_channels, netType, convType):
        super().__init__()
        self.netType = netType

        self.conv_block = DoubleConv(in_channels, out_channels, netType, convType) #, activation='leakyrelu')
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, m, x_skip, m_skip):
        x = self.upsample(x)

        if "partial" in self.netType :
            m = self.upsample(m)

        x = torch.cat([x, x_skip], dim=1)
        # m = torch.cat([m, m_skip], dim=1)

        x, m = self.conv_block(x, m)
        return x, m    
    
class UNet(nn.Module) : 
        
    def __init__(self, in_channels=4,out_channels=3, netType="default", convType="conv2d",doubleLayer=False):
        super().__init__()
                                             
        self.downsample_block_1 = DownSampleBlock(in_channels, 64, netType, convType)
        self.downsample_block_2 = DownSampleBlock(64, 128, netType, convType)
        
        self.middle_conv_block = DoubleConv(128, 256, netType, convType)        
            
        self.upsample_block_2 = UpSampleBlock(128 + 256, 128, netType, convType)
        self.upsample_block_1 = UpSampleBlock(128 + 64, 64, netType, convType)
        
        self.netType = netType
        self.convType = convType
        self.doubleLayer = doubleLayer
        
        if self.doubleLayer : 
            self.downsample_block_1_interm = DownSampleBlock(64, 64, netType, convType)
            self.downsample_block_2_interm = DownSampleBlock(128, 128, netType, convType)
            self.upsample_block_2_interm = UpSampleBlock(256 + 128, 256, netType, convType)
            self.upsample_block_1_interm = UpSampleBlock(128 + 64,  128, netType, convType)
        
        if "conv2d" in self.convType :
            self.last_conv = nn.Conv2d(64, out_channels, 1)
        elif "gated" in self.convType :
            self.last_conv = layer.GatedConv2dWithActivation(64,3,1)
        else : 
            # normalement dilatation=2 mais erreur ici
            self.last_conv = nn.Conv2d(64, out_channels, 1, dilation=1)
        
    def forward(self, x):
        
        if "partial" in self.netType :
            m = ((x[:,-1:] != 0)*1.)
        else :
            m = None
            
        x, m, x_skip1, m_skip1 = self.downsample_block_1(x, m)
        if self.doubleLayer : 
            x, m, x_skip1_interm, m_skip1_interm = self.downsample_block_1_interm(x, m)
            
        x, m, x_skip2, m_skip2 = self.downsample_block_2(x, m)
        if self.doubleLayer : 
            x, m, x_skip2_interm, m_skip2_interm = self.downsample_block_2_interm(x, m)
        
        x, m = self.middle_conv_block(x, m)
        
        if self.doubleLayer : 
            x, m = self.upsample_block_2_interm(x, m, x_skip2_interm, m_skip2_interm) 
        x, m = self.upsample_block_2(x, m, x_skip2, m_skip2) 
        
        if self.doubleLayer : 
            x, m = self.upsample_block_1_interm(x, m, x_skip1_interm, m_skip1_interm)
        x, m = self.upsample_block_1(x, m, x_skip1, m_skip1)
        
        
        out = self.last_conv(x)
        return out
    
    def __str__(self):
        return("UNet({} {})".format(self.netType, self.convType))