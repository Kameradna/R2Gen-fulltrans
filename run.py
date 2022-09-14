from sklearn.model_selection import ParameterGrid
import main_train
import argparse
import torch
import torchvision.models as models

fails = {}
grid_dict = {
    'visual_extractor': ['resnet101','vit_b_16','swin_b'], #,'resnet152','swin_v2_b','vit_l_16','vit_h_14'] #,'wide_resnet50_2','alexnet','regnet_y_16gf','densenet121','convnext_base','efficientnet_v2_l','regnet_y_128gf','resnext101_64x4d'],
    #implement the forward methods in visual_extractor.py and feature size here
    #also implement the transforms tuned to the individual models in
    #also maybe read the papers
    'weights': ['IMAGENET1K_SWAG_E2E_V1',None],#will need to try except for when I fetch the weights
    'monitor_metric': ['CIDEr'],
    'n_gpu': [1],#should I increase batch size in order to speed up training?
    'frozen': [True],
    'cls': [True],
    'repetition': range(4)#probably want 5 of each at least to get a feel
    }


grid = ParameterGrid(grid_dict)
print(f'running {len(grid)} trials at ~6 hours each')

for param in grid:
    print('here we go')
    args = main_train.parse_agrs() #default args are

    if param['visual_extractor'] == 'resnet101':
        d_vf = 2048
    elif param['visual_extractor'] == 'vit_b_16':
        d_vf = 768
    elif param['visual_extractor'] == 'swin_b':
        d_vf = 1024
    else:
        print('get ready for an error caught!')
    args.d_vf = d_vf

    try:
        getattr(models, param['visual_extractor'])(weights=param['weights'])#checking for the swag weights
        weights = 'IMAGENET1K_SWAG_E2E_V1'
    except:
        try:
            getattr(models, param['visual_extractor'])(weights="IMAGENET1K_V2")
            weights = "IMAGENET1K_V2"
        except:
            getattr(models, param['visual_extractor'])(weights="DEFAULT")
            weights="DEFAULT"

    name = f"{param['visual_extractor']}_{weights}_frozen{param['frozen']}_by_{param['monitor_metric']}_{param['repetition']}"

    args.visual_extractor = param['visual_extractor']
    args.weights = weights

    args.monitor_metric = param['monitor_metric']
    args.n_gpu = param['n_gpu']
    args.frozen = param['frozen']
    args.cls = param['cls']

    args.image_dir = 'data/iu_xray/images/'
    args.ann_path = 'data/iu_xray/annotation.json'
    args.dataset_name = 'iu_xray'
    args.max_seq_length = 60
    args.threshold = 3
    args.batch_size = 16
    args.epochs = 100
    args.step_size = 50
    args.gamma = 0.1
    args.early_stop = 100

    args.record_dir = f"recordsruns/records_{name}"
    args.save_dir = f"recordsruns/results_{name}"
    try:
        main_train.main(args)
    except RuntimeError:
        print("Runtime error: CUDA probably out of memory")
        fails[param['visual_extractor']] = 'RuntimeError'
    except NotImplementedError:
        print(f"NotImplemented error: need to provide implementation for {param['visual_extractor']}")
        fails[param['visual_extractor']] = 'Not implemented'
    except KeyboardInterrupt:
        print(f"keyboard interrupt {param['visual_extractor']} individually")
        break
    except:
        print(f"some other reason for failure, need to run {param['visual_extractor']} individually")
        fails[param['visual_extractor']] = 'unknown'

    print("*********Cumulative FAILS*********")
    print(fails)
    print("*********Cumulative FAILS*********")

print("*********ALL FAILS*********")
print(fails)
print("*********ALL FAILS*********")