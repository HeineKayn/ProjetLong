from torchvision.datasets.folder import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import torch

import random

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def downloadDataset():

    for filenumber in reversed(range(0, 458)):
        filenumber = 15
        filenumber = str(filenumber)
        filenumber = "0"*(4-len(filenumber)) + filenumber
        url = "https://samples.vx-underground.org/samples/Blocks/Virusshare%20Collection/Virusshare.{filenumber}.7z"

    # Virusshare.00457.7z

# def getDataset(file:str,factorResize=1,doCrop=True):
#     process = transforms.Compose([
#             transforms.Resize(crop), 
#             transforms.ToTensor()
#     ])
#     return ImageFolder(file, process)

# def getTestImages(file,factorResize=1,doCrop=True,doShuffle=False):
#     dataset = getDataset(file,factorResize,doCrop)
#     dataset = DataLoader(dataset, num_workers=2, batch_size=16, shuffle=doShuffle)
#     return next(iter(dataset))[0]
