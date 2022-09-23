import os
import fnmatch
import pandas as pd
import fnmatch

mega_results = pd.DataFrame()

for root, dir, file in os.walk(os.path.curdir):
    if len(file) > 0:
        if file[0] == "iu_xray.csv":
            file_info = pd.read_csv(f"{root}/{file[0]}")
            run_name = pd.DataFrame([str(root.split("/")[-1].splits('_')[:-1])]*len(file_info.index),columns=['name'])
            file_info = pd.concat([run_name,file_info],axis=1)
            mega_results = pd.concat([mega_results,file_info],axis=0)

print(mega_results)

# #find the mean and std deviation for each name
# mega_results[mega_results['name']]
grouped_results = mega_results.groupby(['name']).mean()


mega_results.to_csv("all_results.csv",index=False)
grouped_results.to_csv("grouped_results.csv")