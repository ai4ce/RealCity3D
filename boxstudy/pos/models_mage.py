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

from util.pos_embed import get_2d_sincos_pos_embed

from oriented_iou_loss import cal_giou

import scipy.stats as stats


class MAGECityGen(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, max_step=32, action_space = 5, mask_ratio_min = 0.5,
                 embed_dim=256, depth=12, num_heads=8, mask_ratio_max=1.0, mask_ratio_mu=0.55, mask_ratio_std=0.25,
                 decoder_embed_dim=64, decoder_depth=4, decoder_num_heads=8,
                 mlp_ratio=4., drop_ratio = 0.1, discre = 50,
                 norm_layer=nn.LayerNorm, device='cuda', activation='sigmoid'):
        super().__init__()
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.device = device
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.activation = activation
        self.max_step =max_step
        self.action_space = action_space
        self.discre = discre

        self.num_heads = num_heads

        if self.activation == 'relu':
            self.activation_fun = F.relu
        elif self.activation == 'sigmoid':
            self.activation_fun = torch.sigmoid

        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - mask_ratio_mu) / mask_ratio_std,
                                                    (mask_ratio_max - mask_ratio_mu) / mask_ratio_std,
                                                    loc=mask_ratio_mu, scale=mask_ratio_std)

        self.fc_embedding =  nn.Sequential(nn.Linear(self.action_space, embed_dim, bias=True))

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, drop_ratio=drop_ratio, attn_drop_ratio=drop_ratio, drop_path_ratio=drop_ratio)#, qk_scale=None
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, drop_ratio=drop_ratio, attn_drop_ratio=drop_ratio, drop_path_ratio=drop_ratio)#, qk_scale=None
            for i in range(decoder_depth)])
        

        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.decoder_pred = nn.Linear(decoder_embed_dim, self.action_space, bias=True) # decoder to patch


        # --------------------------------------------------------------------------

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

    def random_masking(self, x, pos, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  
        _, _, P = pos.shape
        len_keep = L - int(L * mask_ratio)
        
        noise = torch.rand(N, L, device=x.device)  

        ids_shuffle = torch.argsort(noise, dim=1)  
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        ids_masked = ids_shuffle[:, len_keep:]

        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        pos_reserve = torch.gather(pos, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, P))

        x_tar = torch.gather(x, dim=1, index=ids_masked.unsqueeze(-1).repeat(1, 1, D))
        pos_tar = torch.gather(pos, dim=1, index=ids_masked.unsqueeze(-1).repeat(1, 1, P))

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0

        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, pos_reserve, x_tar, pos_tar

    
    def box2corners_th(self, box):
        B = box.size()[0]
        x = box[..., 0:1]
        y = box[..., 1:2]
        w = box[..., 2:3]
        h = box[..., 3:4]
        alpha = box[..., 4:5] # (B, N, 1)
        # print('cos: ',cos)
        x4 = torch.FloatTensor([0.5, -0.5, -0.5, 0.5]).unsqueeze(0).unsqueeze(0).to(box.device) # (1,1,4)
        x4 = x4 * w     # (B, N, 4)
        y4 = torch.FloatTensor([0.5, 0.5, -0.5, -0.5]).unsqueeze(0).unsqueeze(0).to(box.device)
        y4 = y4 * h     # (B, N, 4)
        corners = torch.stack([x4, y4], dim=-1)     # (B, N, 4, 2)
        sin = torch.sin(alpha)
        cos = torch.cos(alpha) 
        row1 = torch.cat([cos, sin], dim=-1)
        row2 = torch.cat([-sin, cos], dim=-1)       # (B, N, 2)
        rot_T = torch.stack([row1, row2], dim=-2)   # (B, N, 2, 2)
        rotated = torch.bmm(corners.view([-1,4,2]), rot_T.view([-1,2,2]))
        rotated = rotated.view([B,-1,4,2])          # (B*N, 4, 2) -> (B, N, 4, 2)
        rotated[..., 0] += x
        rotated[..., 1] += y
        return rotated
    

    def forward_encoder(self, x, pos_embed):
        x= F.relu(self.fc_embedding(x))

        x = x + self.pos_embed_cxy(self.embed_dim, pos_embed)

       

        x = torch.cat([self.mask_token.repeat(x.shape[0], 1, 1), x], dim = 1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward_decoder(self, x, pos_reserve, pos_tar):
        x = self.decoder_embed(x)

        mask_token_decoder = x[:, 0, :].unsqueeze(1)
        x = x[:, 1:, :]

        mask_tokens = mask_token_decoder.repeat(1, pos_tar.shape[1], 1)

        x = torch.cat([x, mask_tokens],  dim = 1)       


        x = x + self.pos_embed_cxy(self.decoder_embed_dim, torch.cat([pos_reserve, pos_tar], dim = 1))

        for blk in self.decoder_blocks:
            x = blk(x)

        x = self.decoder_norm(x)
        x = self.decoder_pred(x)

        out =  torch.cat([self.activation_fun(x[:,:,:4]), x[:,:,4:]],dim=-1)
        
        return out

    def forward(self, imgs, mask_ratio):

        mask_ratio = self.mask_ratio_generator.rvs(1)[0]

        imgs_fea = imgs[:, :, :5]
        position = imgs[:, :, 5:]

        x_reserve, mask, pos_reserve, x_tar, pos_tar = self.random_masking(imgs_fea, position, mask_ratio)
        latent = self.forward_encoder(x_reserve, pos_reserve)

        pred = self.forward_decoder(latent, pos_reserve, pos_tar)  
        pred_box = pred[:, x_reserve.shape[1]:]
        loss_giou, _ = cal_giou(pred_box, x_tar, "pca")
        loss_giou = torch.mean(loss_giou)
        loss_l1 = torch.abs(self.box2corners_th(pred_box)-self.box2corners_th(x_tar)).sum(-1).sum(-1)
        loss_l1 = torch.mean(loss_l1)

        loss = loss_giou + loss_l1

        return loss, pred, mask, loss_giou, loss_l1
    
