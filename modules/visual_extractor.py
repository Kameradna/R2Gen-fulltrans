from unittest.mock import patch
import torch
import torch.nn as nn
import torchvision.models as models


class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.visual_extractor = args.visual_extractor
        self.weights = args.weights
        self.pretrained = args.visual_extractor_pretrained
        model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)#weights
        # model = models.get_model(self.visual_extractor, weights=self.weights)#the modern model registration feature, kameradna
        self.original = args.original
        if args.original == True:
            print("wowza")
            modules = list(model.children())[:-2]
            print(list(model.children())[-2:]) #let's see what we are chopping
            self.model = nn.Sequential(*modules)
            self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
            print(self.model)
            raise(NotImplementedError)
        elif args.original == False:
            print(model)
            model.heads = nn.Identity()
            self.model = model
            print(self.model)
            raise(NotImplementedError)
        else:
            raise(NotImplementedError)

    def forward(self, images):
        patch_feats = self.model(images)

        if self.original == True:
            print(f'patch_feats.shape() = {patch_feats.shape}')
            avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))#averages across the patch features
            print(f'avg_feats.shape() = {avg_feats.shape}')
            batch_size, feat_size, _, _ = patch_feats.shape
            print(f'batch size = {batch_size}, feat_size = {feat_size}')
            patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
            print(f'patch_feats.shape() = {patch_feats.shape}')
            raise(NotImplementedError)

        elif self.original == False:
            print(f'patch_feats.shape() = {patch_feats.shape}')
            avg_feats = patch_feats.squeeze().reshape(-1, patch_feats.size(1)) #likely does nothing to the shape, to be tested
            print(f'avg_feats.shape() = {avg_feats.shape}')
            batch_size, feat_size, _, _ = patch_feats.shape
            print(f'batch size = {batch_size}, feat_size = {feat_size}')
            patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1) #likely does nothing at all since patch feats is unused downstream, and the permute step may be internally done
            print(f'patch_feats.shape() = {patch_feats.shape}')
            raise(NotImplementedError)

        return patch_feats, avg_feats
