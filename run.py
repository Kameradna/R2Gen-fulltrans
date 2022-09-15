from sklearn.model_selection import ParameterGrid
import multiprocessing
from multiprocessing import Process
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

if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')
    fails = {}
    grid_dict = {
        'visual_extractor': ['vit_b_16','resnet101','swin_b'], #,'resnet152','swin_v2_b','wide_resnet50_2','alexnet',regnet_y_16gf','densenet121',]#these work fine on local systems
        # 'visual_extractor': ['vit_l_16','vit_h_14','regnet_y_128gf',]#OOM
        #to be tested:
        # 'visual_extractor': ['convnext_base','efficientnet_v2_l','resnext101_64x4d'],
        #also maybe read the papers
        'weights': ['IMAGENET1K_SWAG_E2E_V1'],
        'monitor_metric': ['CIDEr'],
        'frozen': [True],
        'cls': [True]
        }


    grid = ParameterGrid(grid_dict)
    print(f'running {len(grid)} trials at ~6 hours each')

    for param in grid:
        print('here we go')
        args = main_train.parse_agrs() #default args are

        visfeats_gpu = {
            'resnet101': [2048,1],
            'resnet152':[2048,1],
            'vit_b_16': [768,2],
            'swin_b': [1024,1],
            'swin_v2_b': [1024,1],
            'vit_l_16': [1024,2],#better way to handle attention?
            'vit_h_14': [1280,2],
            'wide_resnet50_2': [2048,1],#still a big question whether the implementation of forward is correct
            'alexnet': [256,1],#big questions about implementation
            'regnet_y_16gf': [3024,1],#big questions about the implementation here is okay
            'regnet_y_128gf': [7392,2],
            'densenet121': [1024,1],
            # 'convnext_base': 'failed from implement',
            # 'efficientnet_v2_l': 'failed from implement',
            # 'resnext101_64x4d': 'failed from implement'
            }
        d_vf, n_gpu_per_model = visfeats_gpu.get(param['visual_extractor'],1)#default 1 feature
        print(d_vf, n_gpu_per_model)
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

        args.visual_extractor = param['visual_extractor']
        args.weights = weights
        args.monitor_metric = param['monitor_metric']
        # args.n_gpu = n_gpu
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

        repetition = 0
        potential_runs = {}
        while repetition < runs:
            for potential in range(int(4/n_gpu_per_model)):
                name = f"{param['visual_extractor']}_{weights}_frozen{param['frozen']}_by_{param['monitor_metric']}_{repetition+offset}"
                repetition += 1
                args.record_dir = f"recordsruns/records_{name}"
                args.save_dir = f"recordsruns/results_{name}"
                indice = int(potential*4/n_gpu_per_model)
                next_indice = int((potential+1)*4/n_gpu_per_model)
                args.use_gpus = f"{indice},{next_indice-1}" if indice != next_indice else f"{indice}"
                print(f"saving as {name}, running on {args.use_gpus}")
                print(f"defining potential run {potential}")
                potential_runs[potential] = Process(target=main_train.main,args=(args,))
                print("MADE IT TO HERE!@!!!!!")

            for specific_run in potential_runs:
                potential_runs[specific_run].start()
                print('run started')

            for specific_run in potential_runs:
                print('Now!')
                potential_runs[specific_run].join()#wait for all to finish
                print('Run done!')
        print("All runs done")