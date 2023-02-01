"""Utilisation : 

    model = SubPixelNetwork(in_channels, upscale_factor)

- **in_channels** : Nombre de channel en entrée
- **upscale_factor** : Facteur d'augmentation de la taille de l'image


---
"""

__all__ = []

import torch.nn as nn

class SubPixelNetwork(nn.Module):
    def __init__(self, in_channels=4, upscale_factor=1):
        super(SubPixelNetwork, self).__init__()
        self.upscale_factor = upscale_factor

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, (upscale_factor ** 2)*3, (3, 3), (1, 1), (1, 1))
        
        # Sub-pixel convolution: rearranges elements in a Tensor of shape (*, r^2C, H, W)
        # to a tensor of shape (C, rH, rW)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x
    
    def __str__(self):
        return("PixelShuffle{}".format(self.upscale_factor))