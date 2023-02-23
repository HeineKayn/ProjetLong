import sys
import projetLib

if len(sys.argv)==2 and sys.argv[1].isdigit() : 
    projetLib.download.downloadAll(int(sys.argv[1]))
elif len(sys.argv)==3 and sys.argv[2].isdigit() : 
    projetLib.download.downloadAll(int(sys.argv[1]),int(sys.argv[2]))