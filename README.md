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

Downloaded resnet101 automatically, this would be helpful for downloading our transformer models directly.

There was some issue with the GPUs not responding, so I made a new bash file to run a different feature extractor.

In R2Gen, the code is such that it uses the deprecated 

I added my own bash file to run transformer-specific visual encoders, and set the gpus to 4. There was no increased utilisation by the baseline program and the transformer encoder naive attempt (just altering --visual_encoder to be vit_b_16) gave us some sort of dimension error. We shall study how the original paper resized the dimensions and how we can alter the process to be more agnostic about the dimensions.


I have added my own print debugs and added a weights field to adhere to modern standards for pytorch 0.15, since the pretrained tag is being deprecated.
The ViT paper uses 768 while the R2Gen paper uses dim of 512. Likely course of action is to reshape the final layer of the ViT architecture or shape the MLP to 512 before feeding directly. Does this violate the goal of 'purely transformer architecture'?

We shall ask Luping.

Luping says fc layer will do it in mapping from feature space to token space.

Also it is apparent from an in-depth reading of the code that the fc features are not used in the encoder and are thus not needed for my ViT interface. Luping also brought up local vs global features in the ViT, I may need to try to find the classification token cls within it somewhere for pretraining goodness.


```shell
images0 shape = torch.Size([16, 3, 224, 224])
images1 shape = torch.Size([16, 3, 224, 224])
images shape = torch.Size([16, 2, 3, 224, 224])
patch_feats.shape() = torch.Size([16, 2048, 7, 7])
avg_feats.shape() = torch.Size([16, 2048])
batch size = 16, feat_size = 2048
patch_feats.shape() = torch.Size([16, 49, 2048])
patch_feats.shape() = torch.Size([16, 2048, 7, 7])
avg_feats.shape() = torch.Size([16, 2048])
batch size = 16, feat_size = 2048
patch_feats.shape() = torch.Size([16, 49, 2048])
fc feats 0.shape = torch.Size([16, 2048]), att_feats 0.shape = torch.Size([16, 49, 2048])
fc feats 1.shape = torch.Size([16, 2048]), att_feats 1.shape = torch.Size([16, 49, 2048])
fc feats.shape = torch.Size([16, 4096]), att_feats.shape = torch.Size([16, 98, 2048])
```
