"""This file aggregates pretraining runs, just edit the grid_dict to select hyperparams or other factors, then run the

python run.py
"""


from sklearn.model_selection import ParameterGrid
import multiprocessing
from multiprocessing import Process
import argparse
import torch
import torchvision.models as models
from copy import deepcopy

parser2 = argparse.ArgumentParser()
parser2.add_argument('--offset', type=int, default=0, help='run offset')
parser2.add_argument('--runs', type=int, default=1, help='runs')
args2 = parser2.parse_args()
offset = args2.offset
runs = args2.runs

import main_train

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    fails = {}
    # grid_dict = {
    #     'visual_extractor': ['vit_b_16', 'resnet101'],
    #     'weights': ['IMAGENET1K_V2'],
    #     'monitor_metric': ['CIDEr'],
    #     'frozen': [False],
    #     'cls': [False],
    #     'lr_ve': [5e-5],#this is impactful?
    #     }

                #to be tried later
        # 'visual_extractor': ['vit_b_32','resnet152','swin_v2_b','wide_resnet50_2','alexnet','regnet_y_16gf','densenet121'],#these work fine on local systems
        # 'visual_extractor': ['vit_l_16','vit_h_14','regnet_y_128gf'],#OOM
        #to be tested:
        # 'visual_extractor': ['convnext_base','efficientnet_v2_l','resnext101_64x4d'],
        #also maybe read the papers

    run_list = [
        #Chexpert cls and frozen for vit
        {'visual_extractor': 'vit_b_16',
        'weights': 'chexpert',
        'monitor_metric': 'CIDEr',
        'frozen': True,
        'cls': True,
        'lr_ve': 0.0},
        #no cls and still frozen
        {'visual_extractor': 'vit_b_16',
        'weights': 'chexpert',
        'monitor_metric': 'CIDEr',
        'frozen': True,
        'cls': False,
        'lr_ve': 0.0},
        #chexpert for resnet, frozen
        {'visual_extractor': 'resnet101',
        'weights': 'chexpert',
        'monitor_metric': 'CIDEr',
        'frozen': True,
        'cls': False,
        'lr_ve': 0.0},


        #Swin_b instead of swin_v2
        {'visual_extractor': 'swin_b',
        'weights': 'IMAGENET1K_V1',
        'monitor_metric': 'CIDEr',
        'frozen': False,
        'cls': False,
        'lr_ve': 5e-5},
        #frozen
        {'visual_extractor': 'swin_b',
        'weights': 'IMAGENET1K_V1',
        'monitor_metric': 'CIDEr',
        'frozen': True,
        'cls': False,
        'lr_ve': 0.0},

        #Vit random inits, cls and frozen
        {'visual_extractor': 'vit_b_16',
        'weights': None,
        'monitor_metric': 'CIDEr',
        'frozen': True,
        'cls': True,
        'lr_ve': 0.0},
        #and random inits, but no cls learning
        {'visual_extractor': 'vit_b_16',
        'weights': None,
        'monitor_metric': 'CIDEr',
        'frozen': False,
        'cls': False,
        'lr_ve': 5e-5},

        #Vit with no cls, lr 0.0001
        {'visual_extractor': 'vit_b_16',
        'weights': 'IMAGENET1K_V1',
        'monitor_metric': 'CIDEr',
        'frozen': False,
        'cls': False,
        'lr_ve': 0.0001},


        #Resnet101	INV1	0.001
        {'visual_extractor': 'resnet101',
        'weights': 'IMAGENET1K_V1',
        'monitor_metric': 'CIDEr',
        'frozen': False,
        'cls': False,
        'lr_ve': 0.001},
        #Resnet101	INV1	0.0001
        {'visual_extractor': 'resnet101',
        'weights': 'IMAGENET1K_V1',
        'monitor_metric': 'CIDEr',
        'frozen': False,
        'cls': False,
        'lr_ve': 0.0001},

    ]

    which_load_visual_extractor = {
            'vit_b_16': 'bit_results_proper/vit_b_16/0.6798918645828945_0.22594534613194003_500bit.pth.tar',
            'resnet101': 'bit_results_proper/resnet101/0.6459725471004509_0.24615514122542928_130bit.pth.tar'
        }


    # grid = ParameterGrid(grid_dict)
    # print(f'running {len(grid)*runs} trials at ~6 hours each')

    # for param in grid:
    for param in run_list:
        args = main_train.parse_agrs() #default args are

        visfeats_gpu = {
            'resnet101': [2048,1],
            'resnet152':[2048,1],
            'vit_b_16': [768,1],
            'vit_b_32': [768,1],
            'swin_b': [1024,2],
            'swin_v2_b': [1024,4],
            # 'vit_l_16': [1024,4],#better way to handle attention?
            # 'vit_h_14': [1280,4],
            'wide_resnet50_2': [2048,1],#still a big question whether the implementation of forward is correct
            'alexnet': [256,1],#big questions about implementation
            'regnet_y_16gf': [3024,1],#big questions about the implementation here is okay
            # 'regnet_y_128gf': [7392,4],
            'densenet121': [1024,1],
            # 'convnext_base': 'failed from implement',
            # 'efficientnet_v2_l': 'failed from implement',
            # 'resnext101_64x4d': 'failed from implement'
            }
        d_vf, n_gpu_per_model = visfeats_gpu.get(param['visual_extractor'],1)#default 1 feature
        args.d_vf = d_vf

        # try:
        #     getattr(models, param['visual_extractor'])(weights=param['weights'])#checking for the swag weights
        #     weights = param['weights']
        # except:
        #     try:
        #         getattr(models, param['visual_extractor'])(weights="IMAGENET1K_V2")
        #         weights = "IMAGENET1K_V2"
        #     except:
        #         getattr(models, param['visual_extractor'])(weights="IMAGENET1K_V1")
        #         weights="IMAGENET1K_V1"

        args.visual_extractor = param['visual_extractor']
        args.weights = param['weights']
        args.monitor_metric = param['monitor_metric']
        # args.n_gpu = n_gpu
        args.frozen = param['frozen']
        print(f"args.frozen is {args.frozen}")
        args.cls = param['cls']

        args.lr_ve = param['lr_ve'] #unimplemented yet

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

        if param['weights'] == 'chexpert':
            args.load_visual_extractor = which_load_visual_extractor[args.visual_extractor]

        repetition = 0
        potential_runs = {}
        potential_runs_args = {}
        potential_func = {}
        while repetition < runs:
            for potential in range(int(4/n_gpu_per_model)):#how many can we potentially run right now?
                name = f"finalruns_{param['visual_extractor']}_{args.weights}_frozen{args.frozen}_cls{args.cls}_lr{args.lr_ve}_by_{args.monitor_metric}_{repetition+offset}"
                repetition += 1
                args.record_dir = f"THEENDISINSIGHT/records_{name}"
                args.save_dir = f"THEENDISINSIGHT/results_{name}"
                indice = int(potential*n_gpu_per_model)
                next_indice = int((potential+1)*n_gpu_per_model)
                args.use_gpus = f"{indice},{next_indice-1}" if indice != next_indice else f"{indice}"
                print(f"saving as {name}, running on {args.use_gpus}")
                # print(f"defining potential run {potential}")
                potential_runs_args[potential] = deepcopy(args)
                potential_func[potential] = deepcopy(main_train.main)
                potential_runs[potential] = Process(target=potential_func[potential],args=(potential_runs_args[potential],))

            for specific_run in potential_runs:
                potential_runs[specific_run].start()

            for specific_run in potential_runs:
                potential_runs[specific_run].join()#wait for all to finish
        print("All runs done of this param set")
    print("All runs done")
