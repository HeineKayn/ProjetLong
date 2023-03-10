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

from dotenv import load_dotenv

load_dotenv()
data_path = os.getenv("data_path")
base = data_path + "/images/"
zipname = "./VirusTemp.7z"

def download_zip(url):
    print("downloading ",url)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        file_size = int(r.headers['Content-Length'])
        chunk_size = 8192
        num_bars = int(file_size / chunk_size)
        t1 = tqdm(r.iter_content(chunk_size=chunk_size), total=num_bars, unit='KB', desc=f"Downloading zip", leave=True, file=sys.stdout)
        with open(zipname, 'wb') as f:
            for chunk in t1:  
                f.write(chunk)

def unzip(url):          
    password = "infected"
    print("extracting ",zipname)
    with py7zr.SevenZipFile(zipname, mode='r', password=password) as z: 
        z.extractall()  
    os.remove(zipname)
    extracted = url.split("/")[-1]
    extracted = ".".join(extracted.split(".")[:-1])
    unzipped  = f"./{extracted}/"
    return unzipped

def extract_features(unzipped,dest):
    entries = os.listdir(unzipped)
    t1 = tqdm(entries, total=len(entries), desc=f"Extracting features", leave=True, file=sys.stdout)
    for entry in t1:
        filepath = unzipped + entry
        try : 
            fileType = subprocess.check_output(f"file {filepath}", shell=True).decode()
            folder = "other"
            if "PE" in fileType    : folder = "pe"
            elif "ELF" in fileType : folder = "elf"
            # elif "MS-DOS" in fileType : folder = "msdos"
            # else : 
            if folder != "other" :
                hashed = str(abs(hash(entry)))
                imgpath = f"{dest}{folder}/{hashed}"
                projetLib.image.extract_img(filepath,imgpath)

        except Exception as e : print(e)
        os.remove(filepath)
    shutil.rmtree(unzipped)

def downloadAll(id,istart=0):
    for i in reversed(range(id,460-3*istart,3)):
        i = str(i)
        i = "0"*(5-len(i)) + i
        dest = f"{base}/Virusshare{i}/"
        if not os.path.exists(dest):
            os.makedirs(dest)
        url = f"https://samples.vx-underground.org/samples/Blocks/Virusshare%20Collection/Virusshare.{i}.7z"

        download_zip(url)
        unzipped = unzip(url)
        extract_features(unzipped,dest)