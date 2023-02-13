import sys
import projetLib

if len(sys.argv)>1 and sys.argv[1].isdigit() : 
    projetLib.download.downloadAll(int(sys.argv[1]))

if len(sys.argv)>2 and sys.argv[1].isdigit() : 
    projetLib.download.downloadAll(int(sys.argv[1]),int(sys.argv[2]))