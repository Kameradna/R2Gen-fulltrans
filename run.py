import os
from sklearn.model_selection import ParameterGrid



repetition = 4

grid_dict = {
    'visual_extractor': ['resnet101','vit_b_16','swin_b'],
    'weights': ['IMAGENET1K_V1'],
    'monitor_metric': ['CIDEr'],
    'n_gpu': ['1'],
    'frozen': ['True'],
    'repetition': range(10)
    }
grid = ParameterGrid(grid_dict)

for param in grid:

    if param['visual_extractor'] == 'resnet101':
        d_vf = 2048
    elif param['visual_extractor'] == 'vit_b_16':
        d_vf = 768
    elif param['visual_extractor'] == 'swin_b':
        d_vf = 1024
    else:
        raise(NotImplementedError)
    command = f" \
    python main_train.py \
    --image_dir data/iu_xray/images/ \
    --ann_path data/iu_xray/annotation.json \
    --dataset_name iu_xray \
    --max_seq_length 60 \
    --threshold 3 \
    --batch_size 16 \
    --epochs 100 \
    --step_size 50 \
    --gamma 0.1 \
    --early_stop 100 \
        \
    --visual_extractor {param['visual_extractor']} \
    --weights {param['weights']} \
    --d_vf {d_vf} \
    --monitor_metric CIDEr \
    --n_gpu 1 \
    --frozen \
    --cls \
        \
    --record_dir recordsnew/recordsfrozen{param['visual_extractor']} \
    --save_dir recordsnew/resultsffrozen{param['visual_extractor']}"

    os.system(command)