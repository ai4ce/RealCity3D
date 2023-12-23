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
from poly_embed import PolyEmbed
from models_autoregress import AutoPoly

import cv2


class MAGECityPolyGen3D(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, max_poly=20, embed_dim=256, depth=12, num_heads=8,  max_bulid = 60, 
                 decoder_embed_dim=64, decoder_depth=4, decoder_num_heads=8, pos_weight = 20,
                 mlp_ratio=4., drop_ratio = 0.1,  discre = 50, device = 'cuda',
                 norm_layer=nn.LayerNorm):
        super().__init__()
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.device = device
        self.max_poly = max_poly
        self.discre = discre
        self.max_bulid = max_bulid

        self.num_heads = num_heads

        self.fc_embedding = PolyEmbed(ouput_dim=embed_dim, device = device)
        self.h_embedding = nn.Linear(1, embed_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, drop_ratio=drop_ratio, attn_drop_ratio=drop_ratio, drop_path_ratio=drop_ratio)#, qk_scale=None
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, drop_ratio=drop_ratio, attn_drop_ratio=drop_ratio, drop_path_ratio=drop_ratio)#, qk_scale=None
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        
        self.decoder_pred = nn.Linear(decoder_embed_dim, decoder_embed_dim, bias=True)
        self.h_pred = nn.Linear(decoder_embed_dim, 1)

        self.midlossfc = nn.Linear(decoder_embed_dim, 2, bias=True)
        # --------------------------------------------------------------------------

        self.automodel = AutoPoly(latent_dim = decoder_embed_dim, device = device)

        self.l1loss = nn.L1Loss()
        self.mseloss = nn.MSELoss()
        self.bceloss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

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
  

    def forward_encoder(self, x, pos, h):
        bsz = x.shape[0]

        x= F.relu(self.fc_embedding(x.flatten(0,1))+self.h_embedding(h.flatten(0,1))).view(bsz, -1, self.embed_dim)

        x = x + self.pos_embed_cxy(self.embed_dim, pos)

        x = torch.cat([self.mask_token.repeat(x.shape[0], 1, 1), x], dim = 1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward_decoder(self, x, posall):
        x = F.relu(self.decoder_embed(x))

        mask_tokens = x[:, 0:1, :]
        x_ = x[:, 1:, :]

        mask_tokens = mask_tokens.repeat(1, self.max_bulid-x_.shape[1], 1)
        x_ = torch.cat([x_, mask_tokens], dim = 1)   

        x_ = x_ + self.pos_embed_cxy(self.decoder_embed_dim, posall)

        for blk in self.decoder_blocks:
            x_ = blk(x_)

        x_ = self.decoder_norm(x_)
        out = self.decoder_pred(x_)
        h_pred = self.h_pred(x_)

        return out, h_pred

    def compute_loss(self, out, polyin, len_tar):
        hyp_bsz = out.shape[0]
        
        poly_out = out[:, :, :2]
        poly_len = out[:, :, 2]
        loss_l1 = self.mseloss(torch.cat([poly_out[i, :len_tar[i]] for i in range(hyp_bsz)], dim = 0),
                               torch.cat([polyin[i, :len_tar[i]] for i in range(hyp_bsz)], dim = 0))

        poly_len_tar = torch.zeros(poly_len.shape).scatter_(1, len_tar.unsqueeze(-1), torch.ones(len(len_tar), 1)).to(self.device)
        loss_len = self.bceloss(poly_len, poly_len_tar)

        return loss_l1, loss_len
    
    
    def forward(self, poly, pos, h, polytar, postar, htar, len_tar):
        ## len_tar object_numpy [[]]
        ## poly_tar [x.shape[0], max_build-remain_num, 20, 2]
        bsz, remain_num, _, _ = poly.shape

        latent = self.forward_encoder(poly, pos, h)

        posall = torch.cat([pos, postar], dim = 1)
        pred_latent, h_pred = self.forward_decoder(latent, posall) 

        loss_height = self.l1loss(torch.cat([h_pred[i, remain_num:remain_num+len(len_tar[i])] for i in range(bsz)], dim=0),
                                  torch.cat([htar[i, :len(len_tar[i])] for i in range(bsz)], dim=0))

        latentautoin = torch.cat([pred_latent[i, remain_num:remain_num+len(len_tar[i])] for i in range(bsz)], dim = 0)
        polyautoin = torch.cat([polytar[i, :len(len_tar[i])] for i in range(bsz)], dim = 0)

        out = self.automodel(latentautoin, polyautoin)

        len_tar = torch.cat([len_tar[i] for i in range(bsz)], dim = 0)
        loss_l1, loss_len = self.compute_loss(out, polyautoin, len_tar) 

        return loss_l1, loss_height, loss_len, out

    def infgen(self, poly, pos, postar):
        bsz, remain_num, _, _ = poly.shape
        assert bsz == 1
        postar_len = postar.shape[1]
        postarin = torch.zeros([1, 60-remain_num, 2]).to(self.device)
        postarin[:, :postar_len, :] = postar

        latent = self.forward_encoder(poly, pos)

        posall = torch.cat([pos, postarin], dim = 1)
        assert posall.shape[1]==self.max_bulid
        pred_latent = self.forward_decoder(latent, posall) 

        latentin = pred_latent[0, remain_num:remain_num+postar_len]
        output_poly = []
        for k in range(latentin.shape[0]): 
            latentin_ = latentin[k:k+1]
            out = self.automodel(latentin_, gen = True)
            inputpoly = out[0, -1, :2].unsqueeze(0).unsqueeze(0)

            for i in range(19):
                out = self.automodel(latentin_, inputpoly)
                if torch.sigmoid(out[0, -1, 2])>0.5:
                        # print('end:', torch.sigmoid(out[0, -1, 2]))
                    break
                inputpoly = torch.cat([inputpoly, out[0, -1, :2].unsqueeze(0).unsqueeze(0)], dim = 1)
            output_poly.append(inputpoly)

        return output_poly
 
    def generate(self, poly, pos, postar, len_tar, img_p):
        bsz, remain_num, _, _ = poly.shape
        assert bsz == 1

        latent = self.forward_encoder(poly, pos)

        posall = torch.cat([pos, postar], dim = 1)
        pred_latent = self.forward_decoder(latent, posall) 

        latentautoin = pred_latent[0, remain_num:remain_num+len(len_tar[0])]
        for i in range(latentautoin.shape[0]):
            latentin = latentautoin[i:i+1]
            point = []
            
            out = self.automodel(latentin, gen = True)
            inputpoly = out[0, -1, :2].unsqueeze(0).unsqueeze(0)
            point.append(np.array(out[0, -1, :2].cpu()))
            step = 1
            for i in range(19):
                out = self.automodel(latentin, inputpoly)
                if torch.sigmoid(out[0, -1, 2])>0.1:
                    # print('end:', torch.sigmoid(out[0, -1, 2]))
                    break
                inputpoly = torch.cat([inputpoly, out[0, -1, :2].unsqueeze(0).unsqueeze(0)], dim = 1)
                point.append(np.array(out[0, -1, :2].cpu()))
                step += 1
            print('step:', step)

            pts = np.array(point, np.int32)
            pts = pts.reshape((-1,1,2)).astype(int)
            cv2.polylines(img_p,[pts],True,(255,0,255),1)

        return img_p
                

