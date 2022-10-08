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
                fileinfo = f"{file}".split("_")
                try:
                    roc = float(fileinfo[0])
                    f1 = float(fileinfo[1])
                    ledger['auroc'].append(roc)
                    ledger['f1'].append(f1)
                except KeyError:
                    print(f"{root}/{file} does not adhere to naming convention")
        print(ledger)
                
                
