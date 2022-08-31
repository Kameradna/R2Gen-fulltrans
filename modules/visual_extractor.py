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
            modules = list(model.children())[:-2]
            self.model = nn.Sequential(*modules)
            self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
  
        elif args.original == False:
            model.heads = nn.Identity()
            self.model = model
        else:
            raise(NotImplementedError)
        self.model.conv_proj.requires_grad_(False)
        self.model.encoder.requires_grad_(False)
        print(self.model)

    def forward(self, images):
        

        if self.original == True:
            patch_feats = self.model(images)
            print(f'patch_feats.shape() = {patch_feats.shape}')# patch_feats.shape() = torch.Size([16, 2048, 7, 7])
            avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))#averages across the patch features
            print(f'avg_feats.shape() = {avg_feats.shape}')# avg_feats.shape() = torch.Size([16, 2048])
            batch_size, feat_size, _, _ = patch_feats.shape
            print(f'batch size = {batch_size}, feat_size = {feat_size}')# batch size = 16, feat_size = 2048
            patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
            print(f'patch_feats.shape() = {patch_feats.shape}')#patch_feats.shape() = torch.Size([16, 49, 2048])

        elif self.original == False:
            # patch_feats = self.model(images)
            # print(f'patch_feats.shape() = {patch_feats.shape}')# patch_feats.shape() = torch.Size([16, 768])
            # avg_feats = patch_feats.squeeze().reshape(-1, patch_feats.size(1)) #likely does nothing to the shape, to be tested
            # print(f'avg_feats.shape() = {avg_feats.shape}')# avg_feats.shape() = torch.Size([16, 768]) confirmed does nothing
            # batch_size, feat_size = patch_feats.shape
            # print(f'batch size = {batch_size}, feat_size = {feat_size}')#batch size = 16, feat_size = 768 the feature size is much lower
            # patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1) #likely does nothing at all since patch feats is unused downstream, and the permute step may be internally done
            # print(f'patch_feats.shape() = {patch_feats.shape}')#patch_feats.shape() = torch.Size([16, 1, 768])

            # #edited from the forward process of ViT
            x = images
            x = self.model._process_input(x)
            n = x.shape[0]

            # Expand the class token to the full batch
            batch_class_token = self.model.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)

            x = self.model.encoder(x)

            ####
            patch_feats = x
            print(f'all_feats.shape() = {patch_feats.shape}')#all_feats.shape() = torch.Size([16, 197, 768])

            #we can try to extract the classification token
            x_star,patch_feats_star = torch.split(patch_feats,split_size_or_sections=[1,196],dim=1)

            print(f'x_star.shape() = {x_star.shape}')
            print(f'patch_feats_star.shape() = {patch_feats_star.shape}')

            ####

            # Classifier "token" as used by standard language architectures
            x = x[:, 0]

            avg_feats = x# avg_feats.shape() = torch.Size([16, 768])
            print(f'avg_feats.shape() = {avg_feats.shape}')
            if x == x_star:
                print('correctly split')
            elif x != x_star:
                print('split not correct')
            raise(NotImplementedError)



        return patch_feats, avg_feats
