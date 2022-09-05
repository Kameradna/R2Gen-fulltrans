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


The planned architecture is such; ((drop in different visual extractors)) --> fc layer reshapes the features to 512 --> R2Gen encoder/decoder


The special sauce is in the att_model.py file with a linear layer implemented to bridge the gap. You need to work out what the deal is with that, any issues it has with passing the global classification tokens, maybe take those out.

Then, you will need to fix the functioning on multi-gpu so that memory is not an issue.

Of special interest is freezing the training of the visual extractor to limit memory usage.

Also definitely time to check on Cider metric and clinical eval metrics.

CIDEr implementation fix worked fine it seems; but the learning of the network is weak. With a frozen transformer, it seems unlikely that we will get strong learning.

Tests-
- Baseline
- Baseline frozen visual extractor (see Frozen Resnet base R2Gen.txt, results and records frozen baseline folders)
- Transformer frozen visual extractor (see currently training)
- Transformer with learning (see ViT paper for finetune settings) (multi-gpu a must?)

To check- 
- is input res a limiting factor for ViT?
- Have I frozen correctly?
- Should I be using a different pretrained model ie. the JFT or other versions?
- the R2Gen paper as is scale to multi-gpu?  (no)
- Batch size key issue for ViT; normal finetune is at 512bs, using BiT hyperrule.
- Also GAP classifier vs cls token usage has quite different performance based on learning rate.


Ran baselines of resnet101 and vit_b_16 with frozen feature extractors, and we will compare the best performances on CIDEr.

Check for use of checkpoints etc, some of the initial epochs often have unexpectedly high CIDEr.

Thoughts on freezing different parts of the architecture and training them independently with various batch sizes?

Currently we are looking to compare the (frozen baseline) to (frozen vit) and also (baseline) to (frozen baseline) which should tell us how much of the gains in R2Gen are from the visual extractor learning vs improved feature representation/feature representation compatibility.
