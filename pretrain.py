from sklearn.model_selection import ParameterGrid
import multiprocessing
from multiprocessing import Process
import argparse
import torchvision.models as models
from copy import deepcopy

#wget http://download.cs.stanford.edu/deep/CheXpert-v1.0-small.zip
#into the data folder
#unzip CheXpert-v1.0-small.zip -d data


def main():
    print("Hello World!")

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    parser2 = argparse.ArgumentParser()
    parser2.add_argument('--offset', type=int, default=0, help='run offset')
    parser2.add_argument('--runs', type=int, default=1, help='runs')
    args2 = parser2.parse_args()
    offset = args2.offset
    runs = args2.runs
    main()