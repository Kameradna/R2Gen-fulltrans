import torch
import torch.nn as nn
import numpy as np

from modules.visual_extractor import VisualExtractor
from modules.encoder_decoder import EncoderDecoder


class R2GenModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(R2GenModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = EncoderDecoder(args, tokenizer)
        self.first_batch_for_debug = True
        # if args.dataset_name == 'iu_xray':
        #     self.forward = self.forward_iu_xray
        # else:
        #     self.forward = self.forward_mimic_cxr

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward(self, images, targets=None, mode='train'): #this was the mother********* line of code that held me back for 2 weeks. The nn.parallel broadcast function could not deal with a forward method with an alias.


        # print(f'images0 shape = {images[:, 0].shape}')# images0 shape = torch.Size([16, 3, 224, 224])
        # print(f'images1 shape = {images[:, 1].shape}')# images1 shape = torch.Size([16, 3, 224, 224])
        
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])#the frontal and side images must be processed separately
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])

        # print(f'att_feats 0.shape = {att_feats_0.shape}')#att_feats 0.shape = torch.Size([16, 49, 2048])
        # print(f'att_feats 1.shape = {att_feats_1.shape}')#att_feats 1.shape = torch.Size([16, 49, 2048])
        fc_feats =torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)

        # if self.first_batch_for_debug:
        #     print(f'att_feats.shape = {att_feats.shape}')#att_feats.shape = torch.Size([16, 98, 2048])
        #     self.first_batch_for_debug = False
        # But for the transformer, we get
        # att_feats 0.shape = torch.Size([16, 1, 768])
        # att_feats 1.shape = torch.Size([16, 1, 768])
        # att_feats.shape = torch.Size([16, 2, 768])

        # Or for when we extract local features, we get
        # att_feats 0.shape = torch.Size([16, 197, 768])
        # att_feats 1.shape = torch.Size([16, 197, 768])
        # att_feats.shape = torch.Size([16, 394, 768])


        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output

    # def forward_mimic_cxr(self, images, targets=None, mode='train'):
    #     att_feats, fc_feats = self.visual_extractor(images)
    #     if mode == 'train':
    #         output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
    #     elif mode == 'sample':
    #         output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
    #     else:
    #         raise ValueError
    #     return output

