import requests 
import shutil
import py7zr
import os
import subprocess

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import requests 
import shutil
import projetLib
from tqdm import tqdm

def download_data(url,dest):
    zipname = "./VirusTemp.7z"
    password = "infected"
     
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(zipname, 'wb') as f:
            for i,chunk in enumerate(r.iter_content(chunk_size=8192)):  
                if i%100 == 0 : print(f"-- chunk downloaded {i}")
                f.write(chunk)
                
    print("extracting ",zipname)
    with py7zr.SevenZipFile(zipname, mode='r', password=password) as z: 
        z.extractall() 
        
    extracted = url.split("/")[-1]
    extracted = extracted.split(".")[:-1]
    unzipped  = f"./{extracted}/"
    os.remove(zipname)
    
    entries = os.listdir(unzipped)
    t1 = tqdm(enumerate(entries), desc=f"Extracting features", colour="#00ff00")
    for i,entry in t1:
        filepath = unzipped + entry
        fileType = subprocess.check_output(f"file {filepath}", shell=True).decode()
    
        folder = "other"
        if "PE" in fileType : folder = "pe"
        elif "ELF" in fileType : folder = "elf"
        elif "MS-DOS" in fileType : folder = "msdos"

        hashed = str(abs(hash(entry)))
        imgpath = f"{dest}{folder}/{hashed}"
        projetLib.data.extract_img(filepath,imgpath)
        t1.set_description(f'Epoch {i + 1}/{len(entries)}')
    shutil.rmtree(unzipped)

def downloadAll(idstart):
    dest = ".projetLib/data/images"
    for i in reversed(range(idstart,460,3)):
        i = str(i)
        i = "0"*(5-len(i)) + i
        url = f"https://samples.vx-underground.org/samples/Blocks/Virusshare%20Collection/Virusshare.{i}.7z"
        print("downloading ",url)
        download_data(url,dest)