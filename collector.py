import os
import fnmatch
import pandas as pd
import fnmatch

mega_results = pd.DataFrame()

name_get={
    'recordsbaseline': 'records_resnet101_IMAGENET1K_V1_frozenFalse_by_CIDEr_x',
    'recordsbaseline2': 'records_resnet101_IMAGENET1K_V1_frozenFalse_by_CIDEr_x',
    'recordsbaseline3':'records_resnet101_IMAGENET1K_V1_frozenFalse_by_CIDEr_x',
    'recordsbaselinenopretrained1':'records_resnet101_None_frozenFalse_by_CIDEr_x',
    'recordsbaselinenopretrained2':'records_resnet101_None_frozenFalse_by_CIDEr_x',
    'recordsfrozenbaseline':'records_resnet101_IMAGENET1K_V1_frozenTrue_by_CIDEr_x',
    'recordsfrozenbaseline2':'records_resnet101_IMAGENET1K_V1_frozenTrue_by_CIDEr_x',
    'recordsfrozenbaseline3':'records_resnet101_IMAGENET1K_V1_frozenTrue_by_CIDEr_x',
    'recordsfrozenbaselinenopretrained':'records_resnet101_None_frozenTrue_by_CIDEr_x',
    'recordsfrozenbaselinenopretrained2':'records_resnet101_None_frozenTrue_by_CIDEr_x',

    'recordsfrozentrans':'records_vit_b_16_IMAGENET1K_V1_frozenTrue_clsFalse_by_CIDEr_x',
    'recordsfrozentrans2':'records_vit_b_16_IMAGENET1K_V1_frozenTrue_clsFalse_by_CIDEr_x',
    'recordsfrozentrans3':'records_vit_b_16_IMAGENET1K_V1_frozenTrue_clsFalse_by_CIDEr_x',
    'recordsfrozentransnopretrained':'records_vit_b_16_None_frozenTrue_clsFalse_by_CIDEr_x',
    'recordsfrozentransnopretrained2':'records_vit_b_16_None_frozenTrue_clsFalse_by_CIDEr_x',
    'recordsfrozentranswithcls': 'records_resnet101_IMAGENET1K_V1_frozenFalse_clsTrue_by_CIDEr_x',
    'recordstrans':'records_vit_b_16_IMAGENET1K_V1_frozenTrue_clsFalse_by_CIDEr_x'
}

for root, dir, file in os.walk(os.path.curdir):
    if len(file) > 0:
        if file[0] == "iu_xray.csv":
            file_info = pd.read_csv(f"{root}/{file[0]}")
            name_eg = root.split("/")[-1]
            if name_eg in name_get:
                name_eg = name_get[name_eg]
            run_name = "".join([name_eg.split('_')[:-1]])
            run_name_series = pd.DataFrame((run_name)*len(file_info.index),columns=['name'])#padding to add to df
            file_info = pd.concat([run_name_series,file_info],axis=1)
            mega_results = pd.concat([mega_results,file_info],axis=0)

print(mega_results)

# #find the mean and std deviation for each name
# mega_results[mega_results['name']]
grouped_results = mega_results[mega_results['best_model_from']=='val'].groupby(['name']).mean()


mega_results.to_csv("all_results.csv",index=False)
grouped_results.to_csv("grouped_val_results.csv")