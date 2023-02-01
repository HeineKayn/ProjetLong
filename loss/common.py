import torch 
import impaintingLib as imp

####################################
# Total Variation loss
####################################

def totalVariation(x):
    """ La Total variation est la mesure de complexité d'une image, elle est utile en combinaison avec une loss perceptuel
    - **x** : torch.Size([batch_size, c, w, h])
    - **return** : int"""
    loss = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + \
            torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return torch.mean(loss)

def keypointLoss(x,x_hat):
    """ Compare les keypoints de l'image originale avec ceux de l'image reconstruite (marche pas très bien)
    - **x** : torch.Size([batch_size, c, w, h])
    - **x_hat** : torch.Size([batch_size, c, w, h])
    - **return** : int"""
    keypointX = imp.components.getKeypoints(x)
    keypointX_hat = imp.components.getKeypoints(x_hat)
    mse = torch.nn.MSELoss()
    loss = mse(keypointX,keypointX_hat)
    if not loss :
        loss = 0
    return loss