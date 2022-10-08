import os
import fnmatch



if __name__ == '__main__':
    this_dir = None

    for root, dir, files in os.walk('.'):
        for file in files:
            if fnmatch.fnmatch(file,"*bit.pth.tar"):
                if root != this_dir:
                    print(f"Entering {root}")
                    ledger = {'auroc': [], 'f1': []}
                this_dir = root
                print(f"{file}".split("_"))
                
                
