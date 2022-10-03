# from sklearn.model_selection import ParameterGrid
# import multiprocessing
# from multiprocessing import Process

import bit_pytorch.models as models

#wget http://download.cs.stanford.edu/deep/CheXpert-v1.0-small.zip
#unzip CheXpert-v1.0-small.zip -d data

from bit_pytorch import train
import bit_common

print("Finished imports, did anything parse?")

def main(args):
    train.main(args)

if __name__ == '__main__':
    parser = bit_common.argparser(models.KNOWN_MODELS.keys())
    parser.add_argument("--datadir", required=True,
                        help="Path to the data folder, preprocessed for torchvision.")
    # parser.add_argument("--annodir", required=True, help="Where are the annotation files to load?")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of background threads used to load data.")
    parser.add_argument("--no-save", dest="save", action="store_false")
    # parser.add_argument("--use_amp", dest="use_amp",action="store_true",
    #                    help="Use Automated Mixed Precision to save potential memory and compute?")
    parser.add_argument("--nnClassCount", type=str, default=120, help="How many classes to train for, more than 5")
    parser.add_argument("--policy", type=str, default='diff', help="Which policy towards u in the dataset do we take?")

    # parser.add_argument("--chexpert", dest="chexpert", action="store_true",help="Run as the chexpert paper?") #could run stomper stuff unedited?
    # parser.add_argument("--pretrained", dest="pretrained", action="store_true",help="Do you want a pretrained network?")
    #args.visual_extractor
    main(parser.parse_args())