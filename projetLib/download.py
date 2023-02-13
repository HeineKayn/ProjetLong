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

import requests 
import shutil
import projetLib

def download_data(url,dest):
    zipname = "./VirusTemp.7z"
    password = "infected"
     
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(zipname, 'wb') as f:
            for i,chunk in enumerate(r.iter_content(chunk_size=8192)):  
                f.write(chunk)
                
    with py7zr.SevenZipFile(zipname, mode='r', password=password) as z: 
        z.extractall() 
        
    extracted = url.split("/")[-1]
    extracted = extracted.split(".")[:-1]
    unzipped  = f"./{extracted}/"
    
    entries = os.listdir(unzipped)
    for entry in entries:
        filepath = unzipped + entry
        fileType = subprocess.check_output(filepath, shell=True).decode()
    
        folder = "other"
        if "PE" in fileType : folder = "pe"
        elif "ELF" in fileType : folder = "elf"
        elif "MS-DOS" in fileType : folder = "msdos"

        hashed = str(abs(hash(extracted)))
        imgpath = f"{dest}{folder}/{hashed}"
        projetLib.data.extract_img(filepath,imgpath)
        
    os.remove(zipname)
    shutil.rmtree(unzipped)

def downloadAll(idstart):
    dest = "../data/images"
    for i in reversed(range(idstart,460,3)):
        i = str(i)
        i = "0"*(5-len(i)) + i
        url = f"https://samples.vx-underground.org/samples/Blocks/Virusshare%20Collection/Virusshare.{i}.7z"
        download_data(url,dest)