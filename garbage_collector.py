import os
import fnmatch

if __name__ == '__main__':
    for root, dir, files in os.walk('.'):
        if fnmatch.fnmatch(files,".pth"):
            for file in files:
                print(file)
