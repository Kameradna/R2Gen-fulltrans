from sklearn.model_selection import ParameterGrid
import argparse
import torch
import torchvision.models as models

parser2 = argparse.ArgumentParser()
parser2.add_argument('--offset', type=int, default=0, help='run offset')
parser2.add_argument('--runs', type=int, default=1, help='runs')
args2 = parser2.parse_args()
offset = args2.offset
runs = args2.runs

import main_train

fails = {}
grid_dict = {
    # 'visual_extractor': ['vit_b_16','resnet101','swin_b','resnet152','swin_v2_b','wide_resnet50_2','alexnet',regnet_y_16gf','densenet121',]#these work fine on local systems
    # 'visual_extractor': ['vit_l_16','vit_h_14','regnet_y_128gf',]#OOM
    'visual_extractor': ['convnext_base','efficientnet_v2_l','resnext101_64x4d'],
    #implement the forward methods in visual_extractor.py and feature size here
    #also implement the transforms tuned to the individual models in
    #also maybe read the papers
    'weights': ['IMAGENET1K_SWAG_E2E_V1'],#will need to try except for when I fetch the weights
    'monitor_metric': ['CIDEr'],
    'n_gpu': [2],#should I increase batch size in order to speed up training?
    'frozen': [True],
    'cls': [True],
    'repetition': range(offset,offset+runs)#probably want 5 of each at least to get a feel
    }


grid = ParameterGrid(grid_dict)
print(f'running {len(grid)} trials at ~6 hours each')

for param in grid:
    print('here we go')
    args = main_train.parse_agrs() #default args are

    visfeats = {
        'resnet101': 2048,
        'resnet152':2048,
        'vit_b_16': 768,
        'swin_b': 1024,
        'swin_v2_b': 1024,
        'vit_l_16': 1024,#better way to handle attention?
        # 'vit_h_14': 1,
        'wide_resnet50_2': 2048,#still a big question whether the implementation of forward is correct
        'alexnet': 256,#big questions about implementation
        'regnet_y_16gf': 3024,#big questions about the implementation here is okay
        'regnet_y_128gf': 7392,
        'densenet121': 1024,
        # 'convnext_base': 'failed from implement',
        # 'efficientnet_v2_l': 'failed from implement',
        # 'resnext101_64x4d': 'failed from implement'
        }
    d_vf = visfeats.get(param['visual_extractor'],1)#default 1 feature
    args.d_vf = d_vf

    try:
        getattr(models, param['visual_extractor'])(weights=param['weights'])#checking for the swag weights
        weights = param['weights']
    except:
        try:
            getattr(models, param['visual_extractor'])(weights="IMAGENET1K_V2")
            weights = "IMAGENET1K_V2"
        except:
            getattr(models, param['visual_extractor'])(weights="IMAGENET1K_V1")
            weights="IMAGENET1K_V1"

    name = f"{param['visual_extractor']}_{weights}_frozen{param['frozen']}_by_{param['monitor_metric']}_{param['repetition']}"
    print(f"saving as {name}")

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
    # try:
    main_train.main(args)
    # except RuntimeError:
    #     print("Runtime error: CUDA probably out of memory")
    #     fails[param['visual_extractor']] = 'RuntimeError'
    # except NotImplementedError:
    #     print(f"NotImplemented error: need to provide implementation for {param['visual_extractor']}")
    #     fails[param['visual_extractor']] = 'Not implemented'
    # except KeyboardInterrupt:
    #     print(f"keyboard interrupt {param['visual_extractor']} individually")
    #     break
    # except:
    #     print(f"some other reason for failure, need to run {param['visual_extractor']} individually")
    #     fails[param['visual_extractor']] = 'unknown'

    print("*********Cumulative FAILS*********")
    print(fails)
    print("*********Cumulative FAILS*********")

print("*********ALL FAILS*********")
print(fails)
print("*********ALL FAILS*********")