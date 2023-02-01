from torchvision.datasets.folder import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import impaintingLib as imp
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

__all__ = ["getDataset","getTestImages","testReal","randomTransfo"]

def getSize(factor=1):
    numWorker = 2 
    wr,hr = resize = (120, 120)
    wc,hc = crop = (64 , 64 )
    batchSize = 32
        
    if factor < 10 : 
        if factor > 1 :
            numWorker = 0
            resize = (wr*factor,hr*factor)
            crop   = (wc*factor,hc*factor)

        if factor > 3 :
            batchSize = 16
    
    # Si on lui passe un gros resize alors c'est qu'on voulait une taille et pas un facteur
    else : 
        crop = (factor,factor)
        batchSize = 3
        
    return resize,crop,numWorker,batchSize

def getMasks(seed=0,resize=1):
    
    path = "data/masks"

    _,crop,numWorker,batchSize = getSize(resize)
    transformations = [
         transforms.Resize(crop), 
         transforms.ToTensor()
        ]

    g = None
    if seed != 0 : 
        g = torch.Generator()
        g.manual_seed(seed)
    
    process = transforms.Compose(transformations)
    dataset = ImageFolder(path, process)
    masks   = DataLoader(dataset, batch_size = batchSize,
                                  shuffle=True, 
                                  generator=g,
                                  num_workers=numWorker)
    return masks

def getDataset(file:str,factorResize=1,doCrop=True):
    """ Crée un objet `ImageFolder` que l'on transforme ensuite en dataloader
    - **file** : Chemin vers un dossier de data 
    - **factorResize** : Les images de ce dataset auront une taille de 64\*factorResize
    - **doCrop** : Si `True` effectue un zoom sur le centre de l'image
    - **return** : ImageFolder"""
    resize = (120*factorResize, 120*factorResize)
    crop   = (64*factorResize, 64*factorResize)
    if doCrop :
        process = transforms.Compose([
             transforms.Resize(resize), 
             transforms.CenterCrop(crop),
             transforms.ToTensor()
        ])
    else : 
        process = transforms.Compose([
             transforms.Resize(crop), 
             transforms.ToTensor()
        ])
    return ImageFolder(file, process)

def getTestImages(file,factorResize=1,doCrop=True,doShuffle=False):
    """ Renvoie un unique batch de taille 16 utile pour tester un modèle
    - **file** : Chemin vers un dossier de data 
    - **factorResize** : Les images de ce dataset auront une taille de 64*factorResize
    - **doCrop** : Si `True` effectue un zoom sur le centre de l'image
    - **doShuffle** Si `True` mélange aléatoirement le dataset, sinon renvoie toujours les premieres images du dossier
    - **return** : torch.Size([16, 3, 64\*factorResize, 64\*factorResize])"""

    dataset = getDataset(file,factorResize,doCrop)
    dataset = DataLoader(dataset, num_workers=2, batch_size=16, shuffle=doShuffle)
    return next(iter(dataset))[0]

def testReal(impainter,base=True,altered=True,segmented=False,keypoints=False,predicted=True):
    """ Va chercher les images, masques et segmentations usuelles présentes dans le dossier **/data/test** pour tester directement un modèle donné dessus
    - **impainter** : Modèle à tester
    - **base** : Si `True` affiche l'image originale
    - **altered** : Si `True` affiche l'image masquée
    - **segmented** : Si `True` affiche l'image segmenté
    - **keypoints** : Si `True` affiche les keypoints
    - **predicted** : Si `True` affiche l'image reconstruite
    - **return** : `None`
    """

    image = getTestImages("./data/test/real",factorResize=2).to(device)
    mask = getTestImages("./data/test/mask",factorResize=2).to(device)
    segment  = getTestImages("./data/test/seg",factorResize=2,doCrop=False).to(device)
    
    imsize = len(image)
    if len(image) < 8 : 
        new_image = image.clone()
        new_mask  = mask.clone()
        new_seg   = segment.clone()
        for i in range(8 - imsize):
            k = random.randint(0, imsize-1)
            new_image = torch.cat((new_image,image[k].unsqueeze(dim=0)),dim=0)
            new_mask = torch.cat((new_mask,mask[k].unsqueeze(dim=0)),dim=0)
            new_seg = torch.cat((new_seg,segment[k].unsqueeze(dim=0)),dim=0)
        image = new_image
        mask = new_mask
        segment = new_seg

    segment = transforms.Grayscale()(segment)
    segment = segment * 255
    segment = (segment / 25) - 1
    segment = torch.round(segment)
    segment = (segment / 9) + 0.1

    mask = transforms.Grayscale()(mask)
    n, c, h, w = image.shape
    x_prime = torch.empty((n, c, h, w), dtype=image.dtype, device=image.device)
    for i, (img, mask) in enumerate(zip(image, mask)):
        propag_img = img.clone()
        mask_bit = (mask > 0.5) * 1.
        for j,channel in enumerate(img[:3]) :
            propag_img[j] = channel * mask_bit
        x_prime[i] = propag_img

    x_prime2 = torch.cat((x_prime,segment),dim=1)
    keypointLayer = imp.components.getKeypoints(image)
    x_prime3 = torch.cat((x_prime2, keypointLayer),dim=1)

    with torch.no_grad():
        image_hat = impainter(x_prime3)
        image_hat = torch.clip(image_hat,0,1)
        image_hat = image_hat[:,:3]

    if base :
        imp.utils.plot_img(image)
    if altered : 
        imp.utils.plot_img(x_prime)
    if segmented : 
        imp.utils.plot_img(segment)
    if keypoints : 
        imp.utils.plot_img(keypointLayer)
    if predicted :
        imp.utils.plot_img(image_hat)

# ------------- DATA AUGMENTATION

from PIL import Image, ImageEnhance
import numpy as np
import math
import random

def zoom(img,factor=0):
    size = (width, height) = (img.size)

    # Si on ne lui donne pas d'arg alors c'est aléatoire
    if factor < 1 :
        (mu,sigma) = (1,3)
        factor = abs(factor)
        factor = np.random.normal(mu, sigma)

    (left, upper, right, lower) = (factor, factor, height-factor, width-factor)
    img = img.crop((left, upper, right, lower))
    img = img.resize(size)
    return img

def crop(img):
    (mu,sigma) = (80,10)
    size = img.size[0]
    factor = np.random.normal(mu, sigma)
    
    transfo = transforms.Compose([transforms.CenterCrop(size*factor/100),
                        transforms.Resize((size,size))])
    
    return transfo(img)

def rotation(img):
    (mu,sigma) = (1.5,1)
    factor = np.random.normal(mu, sigma)
    factor = abs(factor)
    img = img.rotate(factor)
    #img = zoom(img,20)
    return img

def mirror(img):
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

def enhance(img,enhancer):
    (mu,sigma) = (1,0.3)
    factor = np.random.normal(mu, sigma)
    #print(factor)
    img = enhancer.enhance(factor)
    return img

def lumi(img):
    func = ImageEnhance.Brightness(img)
    return enhance(img,func)

def contrast(img):
    func = ImageEnhance.Contrast(img)
    return enhance(img,func)

def color(img):
    func = ImageEnhance.Color(img)
    return enhance(img,func)

def sharpness(img):
    func = ImageEnhance.Sharpness(img)
    return enhance(img,func)

def randomTransfo(imgs):
    """ Fais de la data augmentation (avec une certaine probabilité) sur les imgs données en entrée 
    - **x** : torch.Size([batch_size, **3**, w, h]) 
    - **return** : torch.Size([batch_size, **3**, w, h])
    
    Augmentations implémentés : 
    - crop 
    - rotation
    - mirroir
    - luminosité - *(désactivé)*
    - contraste - *(désactivé)*
    - couleurs - *(désactivé)*
    - sharpness - *(désactivé)*
    """
    
    #(mu,sigma) = (1,0.15)
    (mu,sigma) = (1.3,0.4)
    
    for k,img in enumerate(imgs) : 
        img = transforms.ToPILImage()(img)
        nbTransfo = np.random.normal(mu, sigma)
        nbTransfo = abs(nbTransfo)
        nbTransfo = int(nbTransfo)

        #print(nbTransfo)
        #transfos = [zoom, rotation, mirror, lumi, contrast, color, sharpness]
        #transfos = [mirror, lumi, contrast, color, sharpness]
        transfos = [crop, rotation, mirror]

        for i in range(nbTransfo):
            func = random.choice(transfos)
            #print(func. __name__)
            transfos.remove(func)
            img = func(img)
            
        imgs[k] = transforms.ToTensor()(img.copy())
    return imgs