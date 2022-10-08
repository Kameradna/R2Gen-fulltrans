import os
import fnmatch



if __name__ == '__main__':
    this_dir = None

    for root, dir, files in os.walk('.'):
        if this_dir != root:
            ledger = []
        this_dir = root
        for file in files:
            if fnmatch.fnmatch(file,"*bit.pth.tar"):
                    print(f"{root}/{file}")
