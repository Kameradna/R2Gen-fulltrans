import os
import fnmatch



if __name__ == '__main__':
    this_dir = None

    for root, dir, files in os.walk('.'):
        for file in files:
            if root != this_dir:
                print(f"Entering {root}")
                ledger = []
            this_dir = root
            if fnmatch.fnmatch(file,"*bit.pth.tar"):
                    print("\\\\"+f"{root}/{file}")
