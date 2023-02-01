import impaintingLib as imp
import os 
import torch
from torchvision import transforms

modelPath = "./modelSave/classifierUNet.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

__all__ = ["perceptualClassifier"]

def transformer():
    options = []
    options.append(transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],std=[1/0.229, 1/0.224, 1/0.225]))
    options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transform = transforms.Compose(options)
    return transform

def getTrainedModel():
    model = imp.model.ClassifierUNet().to(device)
    model.load_state_dict(torch.load(modelPath,map_location=device))
    model.eval()
    return model
        
def perceptualClassifier(x, x_hat):
    """ Compare les segmentations de l'originale et de la reconstruction pour les faire correspondre (marche pas très bien)
    - **x** : torch.Size([batch_size, c, w, h])
    - **x_hat** : torch.Size([batch_size, c, w, h])
    - **return** : int"""
    model = getTrainedModel()
    mse = torch.nn.MSELoss()
    
    x = transformer()(x)
    x_hat = transformer()(x_hat)
    
    x_feats = model.getFeatures(x)
    x_hat_feats = model.getFeatures(x_hat)
    
    loss = 0
    loss = mse(x_feats,x_hat_feats)
        
    return loss
    