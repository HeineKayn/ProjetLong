from math import sqrt,ceil
import numpy as np
from PIL import Image
import os

# from torchvision.datasets.folder import ImageFolder
# from torchvision import transforms
# from torch.utils.data import DataLoader
# import torch

# def getImageLoader(file:str,resize,doShuffle=True):
#     process = transforms.Compose([
#             transforms.Resize(resize), 
#             transforms.ToTensor()
#     ])
#     dataset = ImageFolder(file, process)
#     return DataLoader(dataset, num_workers=2, batch_size=16, shuffle=doShuffle)

# def allImageDataset(extensions,resize):
#     datasets = []
#     imgpath = "./data/images/"
#     for folder in os.listdir(imgpath):
#         newpath = imgpath + folder + "/"
#         for ext in extensions :
#             extpath = newpath + ext + "/"
#             if os.path.exists(extpath) : datasets.append(getImageLoader(extpath,resize))
#     return torch.utils.data.ConcatDataset(datasets)

# Serait mieux si file image et renvoie image ?
def crop_img(path,h=256, w=256):
    with Image.open(path) as img:
        img_arr = np.array(img)
        h2,w2 = img_arr.shape
        img_arr = list(np.reshape(img_arr, (h2*w2)))
        img_arr += [0]*(h*w - len(img_arr))
        img_arr = img_arr[:h*w]
        img_arr = np.reshape(np.array(img_arr), (h,w))
        img     = Image.fromarray(img_arr.astype('uint8'), 'L')
        return img

def extract_img(filepath,imagepath, doSave=True):
    with open(filepath, 'rb') as img_set:
        img_arr = list(img_set.read())
        sq   = ceil(sqrt(len(img_arr)))
        rest = (sq*sq)-len(img_arr)
        img_arr += [0]*rest
        
        img_arr = np.array(img_arr)
        img_arr = img_arr.astype('float32')
        #img_arr /= 255
        img_arr = np.reshape(img_arr, (sq,sq))
        img = Image.fromarray(img_arr.astype('uint8'), 'L')
        
        image_directory = "/".join(imagepath.split("/")[:-1])
        if doSave :
            if not os.path.exists(image_directory):
                os.makedirs(image_directory)
            try : img.save(imagepath + ".jpg")
            except Exception as e : print(e)