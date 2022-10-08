import os
import fnmatch



if __name__ == '__main__':
    this_dir = None
    ledger = {'auroc': [], 'f1': []}

    for root, dir, files in os.walk('.'):
        for file in files:
            if fnmatch.fnmatch(file,"*bit.pth.tar"):
                if root != this_dir:
                    if len(ledger['auroc']) > 0:
                        #sort the ledger values and delete the lower 50% of cases
                        ledger['auroc'].sort()
                        centre_value = ledger['auroc'][len(ledger['auroc'])//2]
                        for deletion_candidate in files:
                            if float(deletion_candidate.split("_")[0]) < centre_value:
                                print(f"would remove {deletion_candidate}")
                                #os.remove(f"{this_dir}/{deletion_candidate}")



                    print(f"Entering {root}")
                    ledger = {'auroc': [], 'f1': []}
                this_dir = root
                fileinfo = f"{file}".split("_")
                try:
                    roc = float(fileinfo[0])
                    f1 = float(fileinfo[1])
                    ledger['auroc'].append(roc)
                    ledger['f1'].append(f1)
                except ValueError:
                    print(f"{root}/{file} does not adhere to naming convention")
        
                
                
