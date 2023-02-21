from math import sqrt,ceil
import numpy as np
from PIL import Image,ImageFile
import os

from torchvision.datasets.folder import ImageFolder
from torchvision import transforms
import torch
from torch.utils.data import Subset
from torch.utils.data import DataLoader

from dotenv import load_dotenv

load_dotenv()
data_path = os.getenv("data_path")
imgpath = data_path + "/images/"
benign = "Benign"

ImageFile.LOAD_TRUNCATED_IMAGES = True

def getImageLoader(file:str,resize):
    process = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(resize), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
            #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return ImageFolder(file, process)

def get_malware_dataset(resize,whitelist):
    datasets = []
    for folder in os.listdir(imgpath):
        if folder != benign :
            newpath   = imgpath + folder + "/"
            dataset   = getImageLoader(newpath,resize)
            idwhitelist = [dataset.class_to_idx[x] for x in whitelist if x in dataset.class_to_idx.keys()]
            idx = [i for i in range(len(dataset)) if dataset.imgs[i][1] in idwhitelist]
            for i in range(len(dataset)):
                if dataset.imgs[i][1] in idwhitelist : dataset.imgs[i] = (dataset.imgs[i][0],1)
            dataset = Subset(dataset, idx)
            datasets.append(dataset)
    return torch.utils.data.ConcatDataset(datasets)

def get_benign_dataset(resize):
    datasets = []
    path   = imgpath + benign + "/"
    for folder in os.listdir(path):
        newpath = path + folder + "/"
        dataset = getImageLoader(newpath,resize)
        for i in range(len(dataset)):
            dataset.imgs[i] = (dataset.imgs[i][0],0)
        datasets.append(dataset)
    return torch.utils.data.ConcatDataset(datasets)

def getTrainTest(resize=(224,224),batch_size=32,seed=1,test_proportion=0.2,extensions=["pe","msdos","elf","other"]):
    dataset = get_malware_dataset(resize,extensions)
    lenTrainTest = int(len(dataset)*(1-test_proportion))
    restDataset  = lenTrainTest%batch_size
    g = torch.Generator()
    if seed != 0 :
        g.manual_seed(seed)
    trainDataset,testDataset = torch.utils.data.random_split(dataset, [lenTrainTest-restDataset, len(dataset)-lenTrainTest+restDataset],g)
    benign = get_benign_dataset(resize)
    testDataset = torch.utils.data.ConcatDataset([testDataset,benign])
    return trainDataset,testDataset


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