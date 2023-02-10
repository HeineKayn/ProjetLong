from torchvision.datasets.folder import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import torch

import random

import requests 
import shutil
import py7zr
import os
import subprocess

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def download_data(url,dest):
    zipfile = dest + "VirusTemp.7z"
    password = "infected"
     
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(zipfile, 'wb') as f:
            for i,chunk in enumerate(r.iter_content(chunk_size=8192)):  
                f.write(chunk)
                
    with py7zr.SevenZipFile(zipfile, mode='r', password=password) as z: 
        z.extractall() 
        
    extracted = url.split("/")[-1]
    extracted = extracted.split(".")[:-1]
    unzipped = dest + extracted + "/"
    
    entries = os.listdir(dest)
    for entry in entries:
        filename = unzipped + entry
        fileType = subprocess.check_output(filename, shell=True).decode()
    
        folder = "other"
        if "PE" in fileType : folder = "pe"
        elif "ELF" in fileType : folder = "elf"
        elif "MS-DOS" in fileType : folder = "msdos"

        hashed = hash(extracted)
        img_path = f"{dest}{folder}/{hashed}.png"
        # convert to img
        
    os.remove(zipfile)
    shutil.rmtree(unzipped)

def downloadDataset():

    for filenumber in reversed(range(0, 458)):
        filenumber = 15
        filenumber = str(filenumber)
        filenumber = "0"*(4-len(filenumber)) + filenumber
        url = "https://samples.vx-underground.org/samples/Blocks/Virusshare%20Collection/Virusshare.{filenumber}.7z"
        download_data(url,"../data/images")

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
