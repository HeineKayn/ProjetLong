"""Utilisation : 

    discriminator = imp.model.Discriminator(cnum_in=5, cnum=64)

- **cnum_in** : Nombre de channel en entrée
- **cnum** : Taille de l'image


---
"""

__all__ = []

import torch.nn.functional as F
import torch.nn as nn
import torch

class Conv2DSpectralNorm(nn.Conv2d):
    """Convolution layer that applies Spectral Normalization before every call."""

    def __init__(self, cnum_in,
                 cnum_out, kernel_size, stride, padding=0, n_iter=1, eps=1e-12, bias=True):
        super().__init__(cnum_in,
                         cnum_out, kernel_size=kernel_size,
                         stride=stride, padding=padding, bias=bias)
        self.register_buffer("weight_u", torch.empty(self.weight.size(0), 1))
        nn.init.trunc_normal_(self.weight_u)
        self.n_iter = n_iter
        self.eps = eps

    def l2_norm(self, x):
        return F.normalize(x, p=2, dim=0, eps=self.eps)

    def forward(self, x):

        weight_orig = self.weight.flatten(1).detach()

        for _ in range(self.n_iter):
            v = self.l2_norm(weight_orig.t() @ self.weight_u)
            self.weight_u = self.l2_norm(weight_orig @ v)

        sigma = self.weight_u.t() @ weight_orig @ v
        self.weight.data.div_(sigma)

        x = super().forward(x)

        return x

# ----------------------------------------------------------------------------

class DConv(nn.Module):
    def __init__(self, cnum_in,
                 cnum_out, ksize=5, stride=2, padding='auto'):
        super().__init__()
        padding = (ksize-1)//2 if padding == 'auto' else padding
        self.conv_sn = Conv2DSpectralNorm(
            cnum_in, cnum_out, ksize, stride, padding)
        #self.conv_sn = spectral_norm(nn.Conv2d(cnum_in, cnum_out, ksize, stride, padding))
        self.leaky = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = self.conv_sn(x)
        x = self.leaky(x)
        return x

# ----------------------------------------------------------------------------

class Discriminator(nn.Module):
    def __init__(self, cnum_in, cnum):
        super().__init__()
        self.conv1 = DConv(cnum_in, cnum)
        self.conv2 = DConv(cnum, 2*cnum)
        self.conv3 = DConv(2*cnum, 4*cnum)
        self.conv4 = DConv(4*cnum, 4*cnum)
        self.conv5 = DConv(4*cnum, 4*cnum)
        self.conv6 = DConv(4*cnum, 4*cnum)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = nn.Flatten()(x)

        return x