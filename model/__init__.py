"""Différents modèles sont déjà disponibles dans la librairie mais il est toujours possible d'en ajouter de nouveaux
Actuellement implémentés : 
- UNet (avec différentes convolutions)
- AutoEncoder (un Unet en moins poussé)
- SubpixelNetwork (= pixelShuffle)
- ClassifierUNet (pour la segmentation)
- XceptionNet (pour les keypoints)
- RRDBNet (pour la super résolution)"""

from .autoEncoder import AutoEncoder
from .pixelShuffle import SubPixelNetwork
from .uNet import UNet
from .classifierUNet import ClassifierUNet
from .RRDBNet_arch import RRDBNet
from .keypoint import XceptionNet
from .GAN import Discriminator