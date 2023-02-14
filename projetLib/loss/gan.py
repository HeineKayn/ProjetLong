import torch
import projetLib as proj

__all__ = ["ganLoss"]

def hinge_loss_d(pos, neg):
    """
    gan with hinge loss:
    https://github.com/pfnet-research/sngan_projection/blob/c26cedf7384c9776bcbe5764cb5ca5376e762007/updater.py
    """
    hinge_pos = torch.mean(torch.relu(1-pos))
    hinge_neg = torch.mean(torch.relu(1+neg))
    d_loss = 0.
    d_loss += 0.5*hinge_pos + 0.5*hinge_neg   
    return d_loss

def allButMask(x):
    x_segmented = imp.components.get_segmentation(x, simplify=True, scale_factor=2)
    x_input = torch.cat((x, x_segmented),dim=1)
    x_keypointLayer = imp.components.getKeypoints(x)
    x_input = torch.cat((x_input, x_keypointLayer),dim=1)
    return x_input

def successRate(d):
    n,w = d.shape
    d_red = torch.empty((n), dtype=d.dtype, device=d.device)
    for i,img in enumerate(d) :
        d_red[i] = torch.mean(img)
    res = torch.sum(d_red < 0.5) / torch.numel(d_red)
    return res

def ganLoss(x, x_hat, discriminator):
    """ Loss utilisé par le GAN simpliste que nous avons implémenté
    - **x** : torch.Size([batch_size, c, w, h])
    - **x_hat** : torch.Size([batch_size, c, w, h])
    - **discriminator** : un discriminateur
    - **return** : int"""
    x_input = allButMask(x)
    x_hat_input = allButMask(x_hat)
    
    batch_real_filled = torch.cat((x_input, x_hat_input))
    d_real_gen = discriminator(batch_real_filled)
    d_real, d_gen = torch.split(d_real_gen, 16)
    d_loss = hinge_loss_d(d_real, d_gen)
    
    sr_real = successRate(torch.relu(1-d_real))
    sr_gen  = successRate(torch.relu(1+d_gen))
          
    return d_loss, sr_gen, sr_real