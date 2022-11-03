# R2Gen-fulltrans
R2Gen modified for different feature extractors

Diary entries-
```shell
git clone R2Gen
cd R2Gen
gdown 'https://drive.google.com/u/0/uc?id=1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg&export=download&confirm=t&uuid=5926bb3d-ffc3-4203-b026-4e36a074734d'
unzip -d ./data/ iu_xray.zip

```
We shall see if there is any issues, then we will have some baseline results I hope.

The run only uses one GPU. This is a clear avenue for improvement.

Also there is some error messages about the deprecation of pretrained tag and floordiv.

Setting up environment:
```shell
conda create -n thesis python=3.7 pandas
conda activate thesis
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install opencv-python
pip install gdown

bash run_iu_xray.sh
```

The code downloads the weights and models automatically for you, so look at that.

There was some issue with the GPUs not responding, so I made a new bash file to run a different feature extractor.

In R2Gen, the code is such that it uses deprecated functions and has bad multi-gpu code, it has been patched here. Also, the CIDEr implementation was broken so I fixed it.


Currently, the way to use this repo is to edit run.py. It offers all the saving and run choices available, grid searches, etc. It uses multiprocessing to run multiple runs if it has the excess gpus for it.

For pretraining, edit pretrain.py and run it. To use the pretrained models once you have trained them, edit run.py to load from your specific file and pass 'chexpert' as the weights.
