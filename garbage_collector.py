
"""This file chops all the lower quarter files until there is < 10 files left in each directory"""


import time
import os
import fnmatch

if __name__ == '__main__':
    while True:
        this_dir = None
        this_dir_files = []
        ledger = {'auroc': [], 'f1': []}

        for root, dir, files in os.walk('.'):
            if len(files) > 10:
                for file in files:
                    if fnmatch.fnmatch(file,"*bit.pth.tar"):
                        if root != this_dir:
                            if len(ledger['auroc']) > 0:
                                #sort the ledger values and delete the lower 50% of cases
                                ledger['auroc'].sort()
                                centre_value = ledger['auroc'][len(ledger['auroc'])//4]
                                print(f"roc centre is at {centre_value}")
                                for deletion_candidate in this_dir_files:
                                    if float(deletion_candidate.split("_")[0]) < centre_value:
                                        print(f"Deleting {deletion_candidate}")
                                        os.remove(f"{this_dir}/{deletion_candidate}")
                                
                                ledger['f1'].sort()
                                centre_value = ledger['f1'][len(ledger['f1'])//4]
                                print(f"f1 centre is at {centre_value}")
                                for deletion_candidate in this_dir_files:
                                    try:
                                        if float(deletion_candidate.split("_")[1]) < centre_value:
                                            print(f"Deleting {deletion_candidate}")
                                            os.remove(f"{this_dir}/{deletion_candidate}")
                                    except:
                                        pass

                            print(f"Entering {root}")
                            this_dir = root
                            this_dir_files = []
                            ledger = {'auroc': [], 'f1': []}

                        this_dir_files.append(file)
                        fileinfo = f"{file}".split("_")
                        try:
                            roc = float(fileinfo[0])
                            f1 = float(fileinfo[1])
                            ledger['auroc'].append(roc)
                            ledger['f1'].append(f1)
                        except ValueError:
                            print(f"{root}/{file} does not adhere to naming convention")
        print("Sleeping for 1 minute")
        time.sleep(60)