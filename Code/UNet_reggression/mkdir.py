import os
from pathlib import Path

def mkdirlist(dirlist):
    for i in dirlist:
        if not os.path.exists(i):
            os.makedirs(i)
