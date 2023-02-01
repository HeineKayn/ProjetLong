"""Utilisation : 

    model = AutoEncoder(in_channels)

---
"""

import torch.nn as nn

__all__ = []

class AutoEncoder(nn.Module):
    
    def __init__(self,in_channels=5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d( 128, 64, 4, 2, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d( 64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d( 32, 3, 4, 2, 1, bias=False),
        )
        
    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y
    
    def __str__(self):
        return("AutoEncoder")