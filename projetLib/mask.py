import numpy as np
import torch
import impaintingLib as imp

from torchvision import transforms
from PIL import Image
from os import listdir
from os.path import isfile, join
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def propagate(imgs,masks):
    """Propage un masque généré sur toutes les couche d'une image RGB
    - **imgs** : torch.Size([batch_size, **3**, w, h])
    - **masks** : torch.Size([batch_size, **1**, w, h])
    - **return** : torch.Size([batch_size, **3**, w, h])
    """

    n, c, h, w = imgs.shape     # c+1
    imgs_masked = torch.empty((n, c, h, w), dtype=imgs.dtype, device=imgs.device)
    for i, (img, mask) in enumerate(zip(imgs, masks)):
        propag_img = img.clone()
        mask_bit = (mask < 0.5) * 1.
        for j,channel in enumerate(img[:3]) :
            propag_img[j] = channel * mask_bit

        #imgs_masked[i] = torch.cat((propag_img,mask),0)
        imgs_masked[i] = propag_img

    return imgs_masked

class Alter :

    def __init__(self, min_cut=15, max_cut=45, seed=0, resize=1):
        self.min_cut = min_cut
        self.max_cut = max_cut
        """Mincut et Maxcut conditionnent la position des masques carrés. Si les deux sont à 0 il peut y'avoir un carré noir partout sur l'image mais si les deux sont à 50 il ne pourra pas y avoir de carré sur les bords par exemple"""
        self.seed    = seed
        """Si la seed est à 0 alors le masque sera aléatoire. Si elle est différente de 0 alors une même seed donnera toujours le même résultat"""
        self.resize = resize
        """Coefficient de taille de l'image. La taille des masques sera toujours resize*64"""
        
        self.maskLoader = imp.data.getMasks(resize=resize,seed=seed)
        self.maskIter   = iter(self.maskLoader)
    
    # Generate square mask
    def squareMask(self,imgs):
        """Crée un masque carré aléatoire et le propage sur l'image
        - **imgs** : torch.Size([batch_size, **3**, w, h])
        - **masks** : torch.Size([batch_size, **1**, w, h])
        - **return** : torch.Size([batch_size, **3**, w, h])
        """
        
        if self.seed != 0:
            np.random.seed(self.seed)
        
        n, c, h, w = imgs.shape
        w1 = np.random.randint(0, w, n)
        h1 = np.random.randint(0, w, n)
        
        w2 = np.random.randint(0, w, n)
        h2 = np.random.randint(0, w, n)
        
        masks = torch.empty((n, 1, h, w), dtype=imgs.dtype, device=imgs.device)
        for i, (img, w11, h11, w22, h22) in enumerate(zip(imgs, w1, h1, w2, h2)):
            cut_img = torch.full((1,h,w),0, dtype=img.dtype, device=img.device)
            cut_img[:, h11:h11 + h11, w11:w11 + w11] = 1
            cut_img[:, h22:h22 + h11, w22:w22 + w22] = 1
            masks[i] = cut_img
            
        imgs_masked = propagate(imgs,masks)
        return imgs_masked
    
    def fullMask(self,imgs):
        """Remplace chaque pixel de l'image par du noire
        - **imgs** : torch.Size([batch_size, **3**, w, h])
        - **return** : torch.Size([batch_size, **3**, w, h])
        """
        n, c, h, w = imgs.shape
        masks = torch.full((n, 1, h, w),1, dtype=imgs.dtype, device=imgs.device)
        imgs_masked = propagate(imgs,masks)
        return imgs_masked
    
    def downScale(self,imgs, scale_factor=2, upscale=True):
        """Baisse la résolution de l'image
        - **imgs** : torch.Size([batch_size, **3**, w, h])
        - **scale_factor** : En fonction de la taille de l'image
        - **return** : torch.Size([batch_size, **3**, w, h])
        """
        imgs_low = torch.nn.MaxPool2d(kernel_size=scale_factor)(imgs)
        if upscale:
            imgs_low = torch.nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)(imgs_low)
        return imgs_low
    
    def irregularMask(self,imgs):
        """Récupère des masques irréguliers dans le dossier /data/masks et les applique sur les images
        - **imgs** : torch.Size([batch_size, **3**, w, h])
        - **return** : torch.Size([batch_size, **3**, w, h])
        """
        
        try:
            masks,_ = next(self.maskIter)
        except StopIteration:
            self.maskIter = iter(self.maskLoader)
            masks,_ = next(self.maskIter)
            
        masks = masks[:,:1].to(device)
        imgs_masked = propagate(imgs,masks)
        return imgs_masked


# Generate random mask