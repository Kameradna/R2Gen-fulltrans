from unittest.mock import patch
import torch
import torch.nn as nn
import torchvision.models as models
import fnmatch
from multiprocessing import Lock


class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.visual_extractor = args.visual_extractor
        self.weights = args.weights
        if args.cls:
            self.cls = args.cls
            # print("using cls token if available")
        self.printfirst = True
        # print(f"weights are {self.weights}")
        # print(f"d_vf is {args.d_vf}")
        model = getattr(models, args.visual_extractor)(weights=args.weights)
        if fnmatch.fnmatch(self.visual_extractor,"*resnet*"):
            modules = list(model.children())[:-2]
            self.model = nn.Sequential(*modules)
            self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

        elif fnmatch.fnmatch(self.visual_extractor,"vit*"):
            self.model = model
        elif fnmatch.fnmatch(self.visual_extractor,"swin*"):
            model.head = nn.Identity()
            self.model = model
        elif fnmatch.fnmatch(self.visual_extractor,"alexnet"):
            modules = list(model.children())[:-2]
            self.model = nn.Sequential(*modules)
        elif fnmatch.fnmatch(self.visual_extractor,"regnet*"):
            modules = list(model.children())[:-2]
            self.model = nn.Sequential(*modules)
        elif fnmatch.fnmatch(self.visual_extractor,"densenet*"):
            self.model = model
        else:
            print(f"we have not implemented the {self.visual_extractor} visual extractor for this paper")
            raise(NotImplementedError)
        if args.frozen == True:
            self.model.requires_grad_(False)
            # print("We are freezing the visual extractor.")
        # print(self.model)


    def forward(self, images):
        
        if fnmatch.fnmatch(self.visual_extractor,"*resnet*"):
            patch_feats = self.model(images)
            # print(f"feats.shape() = {patch_feats.shape}")
            avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))#averages across the patch features
            batch_size, feat_size, _, _ = patch_feats.shape
            patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
            # print(f"feats.shape() = {patch_feats.shape}")

        elif fnmatch.fnmatch(self.visual_extractor,"vit*"):

            x = self.model._process_input(images)
            n = x.shape[0]
            batch_class_token = self.model.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)
            x = self.model.encoder(x)
            patch_feats = x
            if not self.cls:
                x_star,patch_feats_star = torch.split(patch_feats,split_size_or_sections=[1,196],dim=1)
                x_star = x_star[:, 0]
            elif self.cls:
                patch_feats_star = patch_feats
            else:
                raise(NotImplementedError)
            avg_feats = patch_feats_star
            patch_feats = patch_feats_star

        elif fnmatch.fnmatch(self.visual_extractor,"swin*"):
            x = self.model.features(images)
            x = self.model.norm(x)#who knows whether this helps?
            batch_size, _, _, feat_size = x.shape #changed from the vit section
            x = x.reshape(batch_size, feat_size, -1).permute(0,2,1)
            patch_feats = x
            avg_feats = patch_feats
        
        elif fnmatch.fnmatch(self.visual_extractor,"alexnet"):
            patch_feats = self.model(images)
            batch_size, feat_size, _, _ = patch_feats.shape
            patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
            avg_feats = patch_feats

        elif fnmatch.fnmatch(self.visual_extractor,"regnet*"):
            patch_feats = self.model(images)
            # print(f"feats.shape() = {patch_feats.shape}")
            # avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))#averages across the patch features
            # print(f"avg.shape() = {avg_feats.shape}")
            batch_size, feat_size, _, _ = patch_feats.shape
            patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
            # print(f"feats.shape() = {patch_feats.shape}")
            avg_feats = patch_feats
        
        elif fnmatch.fnmatch(self.visual_extractor,"densenet*"):
            x = self.model.features(images)
            # print(f"feats.shape() = {x.shape}")

            batch_size, feat_size, _, _ = x.shape #changed from the vit section
            x = x.reshape(batch_size, feat_size, -1).permute(0,2,1)
            patch_feats = x
            avg_feats = patch_feats

        else:
            print(f"you should implement the forward method for {self.visual_extractor}")
            raise(NotImplementedError)

        # if self.printfirst:
        #     self.printfirst = False
        #     print(f"feats.shape() = {patch_feats.shape}")
        return patch_feats, avg_feats
