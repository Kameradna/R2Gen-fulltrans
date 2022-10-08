# from sklearn.model_selection import ParameterGrid
# import multiprocessing
# from multiprocessing import Process

import bit_pytorch.models as models

#wget http://download.cs.stanford.edu/deep/CheXpert-v1.0-small.zip
#unzip CheXpert-v1.0-small.zip -d data

from bit_pytorch import train
import bit_common

print("Finished imports, did anything parse?")

# python pretrain.py --name bit_proper_lr0.00005 --datadir data/CheXpert-v1.0-small --dataset CheXpert --eval_every 5 --logdir bit_proper_lr0.00005 --batch_split 4

def main(args):
    fails = []
    list_vis_ext = ['densenet121', 'vit_b_16', 'resnet101', 'swin_v2_b']
    for visual_extractor in list_vis_ext:
        args.name = visual_extractor
        args.visual_extractor = visual_extractor
        train.main(args)
        # print(f"Will need to rerun {visual_extractor}")
        # fails.append(visual_extractor)
    for fail in fails:
        print(f"Need to rerun {fail}")

if __name__ == '__main__':
    parser = bit_common.argparser(models.KNOWN_MODELS.keys())
    parser.add_argument("--datadir", required=True,
                        help="Path to the data folder, preprocessed for torchvision.")
    # parser.add_argument("--annodir", required=True, help="Where are the annotation files to load?")
    # parser.add_argument("--visual_extractor", type=str, required = True, help="Which visual extractor would you like to train?")
    parser.add_argument("--weights", type=str, default='IMAGENET1K_V1', help="Which initial weights would you like to use?")
    parser.add_argument("--optim", type=str, default='SGD', help="Which optimser to use?")

    parser.add_argument("--workers", type=int, default=8,
                        help="Number of background threads used to load data.")
    parser.add_argument("--no-save", dest="save", action="store_false")
    # parser.add_argument("--use_amp", dest="use_amp",action="store_true",
    #                    help="Use Automated Mixed Precision to save potential memory and compute?")
    parser.add_argument("--nnClassCount", type=int, default=14, help="How many classes to train for, more than 5")
    parser.add_argument("--policy", type=str, default='diff', help="Which policy towards u in the dataset do we take?")

    # parser.add_argument("--chexpert", dest="chexpert", action="store_true",help="Run as the chexpert paper?") #could run stomper stuff unedited?
    # parser.add_argument("--pretrained", dest="pretrained", action="store_true",help="Do you want a pretrained network?")
    #args.visual_extractor
    main(parser.parse_args())