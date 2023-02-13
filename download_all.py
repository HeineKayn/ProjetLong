import sys
import projetLib

if len(sys.argv)>1 and sys.argv[1].isdigit() : 
    projetLib.download.downloadAll(int(sys.argv[1]))