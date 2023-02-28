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

def getImageLoader(file:str,resize,doRGB,doCrop):
    if doCrop : func = Crop_img
    else : func = Resize_img
    process = transforms.Compose([
            transforms.Grayscale(),
            # transforms.Resize(resize), 
            func(resize,doRGB),
            transforms.ToTensor()
            #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return ImageFolder(file, process)

def get_malware_dataset(resize,whitelist,doRGB,doCrop):
    datasets = []
    for folder in os.listdir(imgpath):
        if folder != benign :
            newpath   = imgpath + folder + "/"
            dataset   = getImageLoader(newpath,resize,doRGB,doCrop)
            idwhitelist = [dataset.class_to_idx[x] for x in whitelist if x in dataset.class_to_idx.keys()]
            idx = [i for i in range(len(dataset)) if dataset.imgs[i][1] in idwhitelist]
            for i in range(len(dataset)):
                if dataset.imgs[i][1] in idwhitelist : dataset.imgs[i] = (dataset.imgs[i][0],1)
            dataset = Subset(dataset, idx)
            datasets.append(dataset)
    return torch.utils.data.ConcatDataset(datasets)

def get_benign_dataset(resize, whitelist, doRGB, doCrop):
    datasets = []
    path   = imgpath + benign + "/"
    for folder in os.listdir(path):
        newpath = path + folder + "/"
        dataset   = getImageLoader(newpath,resize,doRGB,doCrop)
        idwhitelist = [dataset.class_to_idx[x] for x in whitelist if x in dataset.class_to_idx.keys()]
        idx = [i for i in range(len(dataset)) if dataset.imgs[i][1] in idwhitelist]
        for i in range(len(dataset)):
            if dataset.imgs[i][1] in idwhitelist : dataset.imgs[i] = (dataset.imgs[i][0],0)
        dataset = Subset(dataset, idx)
        datasets.append(dataset)
    return torch.utils.data.ConcatDataset(datasets)

def getTrainTest(resize=(224,224),batch_size=32,seed=1,trainSize=[],testSize=[],extensions=["pe","msdos","elf","other"],doRGB=False,doCrop=True):
    g = torch.Generator()
    if seed != 0 : g.manual_seed(seed)

    malTrain,benTrain = [],[]
    malTest,benTest   = [],[]
    for ext in extensions:
        malExt = get_malware_dataset(resize,[ext],doRGB,doCrop) 
        malExt,restMal = torch.utils.data.random_split(malExt, [trainSize[0]//len(extensions),len(malExt)-trainSize[0]//len(extensions)],g)
        malTrain.append(malExt)
        restMal,_ = torch.utils.data.random_split(restMal, [testSize[0]//len(extensions),len(restMal)-testSize[0]//len(extensions)],g)
        malTest.append(restMal)

        benExt = get_benign_dataset(resize,[ext],doRGB,doCrop)
        benExt,restBen = torch.utils.data.random_split(benExt, [trainSize[1]//len(extensions),len(benExt)-trainSize[1]//len(extensions)],g)
        benTrain.append(benExt)
        restBen,_ = torch.utils.data.random_split(restBen, [testSize[1]//len(extensions),len(restBen)-testSize[1]//len(extensions)],g)
        benTest.append(restBen)

    # print("Répartition Malware train -----")
    # for x in malTrain : print(len(x))
    # print("Répartition Benin train -----")
    # for x in benTrain : print(len(x))
    # print("Répartition Malware test -----")
    # for x in malTest  : print(len(x))
    # print("Répartition Benin test -----")
    # for x in benTest  : print(len(x))

    malTrain = torch.utils.data.ConcatDataset(malTrain)
    benTrain = torch.utils.data.ConcatDataset(benTrain)
    malTest  = torch.utils.data.ConcatDataset(malTest)
    benTest  = torch.utils.data.ConcatDataset(benTest)
    trainDataset = torch.utils.data.ConcatDataset([malTrain,benTrain])
    testDataset  = torch.utils.data.ConcatDataset([malTest ,benTest])
    return trainDataset,testDataset

class Crop_img(torch.nn.Module):

    def __init__(self, size, doRGB=False):
        super().__init__()
        self.size = size
        self.doRGB = doRGB
        
    def crop_img(self, img, resize):
        h,w = resize
        img_arr = np.array(img)
        h2,w2   = img_arr.shape
        img_arr = list(np.reshape(img_arr, (h2*w2)))
        img_arr += [0]*(h*w - len(img_arr))
        img_arr = img_arr[:h*w]
        img_arr = np.reshape(np.array(img_arr), (h,w))
        img     = Image.fromarray(img_arr.astype('uint8'), 'L')
        return img

    def crop_img_RGB(self, img, resize):
        h,w = resize
        img_arr = np.array(img)
        h2,w2   = img_arr.shape
        img_arr = list(np.reshape(img_arr, (h2*w2)))
        img_arr += [0]*(h*w*3 - len(img_arr))
        img_arr = img_arr[:h*w*3]
        img_arr = np.reshape(np.array(img_arr), (h,w,3))
        img     = Image.fromarray(img_arr.astype('uint8'), 'RGB')
        return img

    def forward(self, img):
        if self.doRGB : return self.crop_img_RGB(img,self.size)
        else : return self.crop_img(img,self.size)

class Resize_img(torch.nn.Module):

    def __init__(self, size, doRGB=False):
        super().__init__()
        self.size = size
        self.doRGB = doRGB
        
    def resize_img(self, img, resize):
        return transforms.Resize(resize)(img)

    def resize_img_RGB(self, img, resize):
        img_arr = np.array(img)
        h2,w2   = img_arr.shape
        img_arr = list(np.reshape(img_arr, (h2//3,w2//3,3)))
        img     = Image.fromarray(img_arr.astype('uint8'), 'RGB')
        return transforms.Resize(resize)(img)

    def forward(self, img):
        if self.doRGB : return self.resize_img(img,self.size)
        else : return self.resize_img_RGB(img,self.size)

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