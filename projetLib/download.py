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
import sys

def download_data(url,dest):
    zipname = "./VirusTemp.7z"
    password = "infected"

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        file_size = int(r.headers['Content-Length'])
        chunk_size = 8192
        num_bars = int(file_size / chunk_size)
        t1 = tqdm(r.iter_content(chunk_size=chunk_size), total=num_bars, unit='KB', desc=f"Downloading zip", leave=True, file=sys.stdout)
        with open(zipname, 'wb') as f:
            for chunk in t1:  
                f.write(chunk)
                
    print("extracting ",zipname)
    with py7zr.SevenZipFile(zipname, mode='r', password=password) as z: 
        z.extractall() 
        
    extracted = url.split("/")[-1]
    extracted = ".".join(extracted.split(".")[:-1])
    unzipped  = f"./{extracted}/"
    os.remove(zipname)
    
    entries = os.listdir(unzipped)
    t1 = tqdm(entries, total=len(entries), desc=f"Extracting features", leave=True, file=sys.stdout)
    for entry in t1:
        filepath = unzipped + entry
        try : 
            fileType = subprocess.check_output(f"file {filepath}", shell=True).decode()
            folder = "other"
            if "PE" in fileType : folder = "pe"
            elif "ELF" in fileType : folder = "elf"
            elif "MS-DOS" in fileType : folder = "msdos"

            hashed = str(abs(hash(entry)))
            imgpath = f"{dest}{folder}/{hashed}"
            projetLib.data.extract_img(filepath,imgpath)
        except Exception as e : print(e)
    
        
    shutil.rmtree(unzipped)

def downloadAll(id,istart=0):
    base = "./data/images"
    for i in reversed(range(id+3*istart,460,3)):
        i = str(i)
        i = "0"*(5-len(i)) + i
        dest = f"{base}/Virusshare{i}/"
        if not os.path.exists(dest):
            os.makedirs(dest)
        url = f"https://samples.vx-underground.org/samples/Blocks/Virusshare%20Collection/Virusshare.{i}.7z"
        print("downloading ",url)
        download_data(url,dest)