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
from timm.models.vision_transformer import Block
from util.transformer import Block
from reformer_pytorch import Reformer
import time


from poly_embed import PolyEmbed

import cv2



class MAGECityPolyGen(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, max_poly=20, embed_dim=256, depth=12, num_heads=8,  max_build = 60, 
                 decoder_embed_dim=64, mlp_ratio=4., drop_ratio = 0.1,  discre = 50, device = 'cuda',
                 norm_layer=nn.LayerNorm):
        super().__init__()
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.device = device
        self.max_poly = max_poly
        self.discre = discre
        self.max_build = max_build
        self.fix_mask_token = False

        self.num_heads = num_heads

        self.fc_embedding = PolyEmbed(ouput_dim=embed_dim, device = device)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim if self.fix_mask_token else embed_dim))

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, drop_ratio=drop_ratio, attn_drop_ratio=drop_ratio, drop_path_ratio=drop_ratio)#, qk_scale=None
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, 15, bias=True)

        self.crossentropyloss = nn.CrossEntropyLoss()

        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def pos_embed_cxy(self, embed_dim, pos):
        assert embed_dim % 2 == 0
        position = pos.cpu().numpy()
        emb_h = self.get_1d_embed(embed_dim // 2, position[:, :, 0])  # (H*W, D/2)
        emb_w = self.get_1d_embed(embed_dim // 2, position[:, :, 1])  # (H*W, D/2)

        emb = np.concatenate([emb_h, emb_w], axis=2) # (H*W, D)
        emb = torch.tensor(emb).to(self.device)
        return emb

    def get_1d_embed(self, embed_dim, pos):
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=np.float32)
        omega /= embed_dim / 2.
        omega = 1. / 10000**omega  # (D/2,)
        batch_n, num_b = pos.shape

        pos = pos.reshape(-1)  # (M,)
        out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product
        out = out.reshape(batch_n, num_b, embed_dim // 2)

        emb_sin = np.sin(out) # (M, D/2)
        emb_cos = np.cos(out) # (M, D/2)

        emb = np.concatenate([emb_sin, emb_cos], axis=2)  # (M, D)
        return emb
    

    def forward(self, x, pos, tarclass):
        bsz = x.shape[0]

        x= F.relu(self.fc_embedding(x.flatten(0,1))).view(bsz, -1, self.embed_dim)

        x = x + self.pos_embed_cxy(self.embed_dim, pos)

        x = torch.cat([self.mask_token.repeat(x.shape[0], 1, 1), x], dim = 1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        x = self.decoder_embed(x[:, 0])
        
        loss = self.crossentropyloss(x, tarclass)

        acc = 1 - torch.count_nonzero(torch.argmax(x, dim = -1) - tarclass)/bsz
        return loss, acc

    