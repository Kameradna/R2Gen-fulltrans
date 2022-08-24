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
conda create -n thesis python=3.7
conda activate thesis
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch=1.7.1 torchvision=0.8.2 --yes
pip install gdown pandas


bash run_iu_xray.sh
