# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
# from timm.models.vision_transformer import Block
# from util.transformer import Block
# from reformer_pytorch import Reformer
import time
import sys

sys.path.append('../2dpoly')
from models_mage import MAGECityPolyGen
# from poly_embed import PolyEmbed

# import cv2



class MAGECityPolyGenProb(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self,  device = 'cuda',):
        super().__init__()
        # --------------------------------------------------------------------------

        self.mae = MAGECityPolyGen(drop_ratio = 0.1, num_heads=8, device = device, max_build=60,
                        depth=12, embed_dim=512, decoder_embed_dim = 512, discre=50,
                        decoder_depth=8, decoder_num_heads=8, pos_weight = 100)
        # self.mae.eval()
        
        # self.decoder = nn.Linear(512*60,  512, bias=True)
        self.decoder_embed = nn.Linear(512, 15, bias=True)
        self.crossentropyloss = nn.CrossEntropyLoss()

        # self.initialize_weights()

    # def initialize_weights(self):
    #     self.apply(self._init_weights)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         # we use xavier_uniform following official JAX ViT:
    #         torch.nn.init.xavier_uniform_(m.weight)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    
    def forward(self, x, pos, tarclass):
        bsz = x.shape[0]

        latent = self.mae.forward_encoder(x, pos)

        # x = self.decoder_embed(F.relu(self.decoder(latent[:, 1:].flatten(1,2))))
        x = self.decoder_embed(latent[:, 0])
        
        loss = self.crossentropyloss(x, tarclass)

        acc = 1 - torch.count_nonzero(torch.argmax(x, dim = -1) - tarclass)/bsz
        
        return loss, acc

    