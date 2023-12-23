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

from util.pos_embed import get_2d_sincos_pos_embed
from poly_embed import PolyEmbed
from models_autoregress import AutoPoly




class MAGECityPolyGen(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, max_poly=20, embed_dim=256, depth=12, num_heads=8,  max_build = 60, 
                 decoder_embed_dim=64, decoder_depth=4, decoder_num_heads=8, pos_weight = 20,
                 mlp_ratio=4., drop_ratio = 0.1,  discre = 50, device = 'cuda', max_road_len = 38, 
                 norm_layer=nn.LayerNorm):
        super().__init__()
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.append = True
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.device = device
        self.max_poly = max_poly
        self.discre = discre
        self.max_build = max_build

        self.num_heads = num_heads

        self.fc_embedding = PolyEmbed(ouput_dim=embed_dim, device = device)
        self.road_embedding = PolyEmbed(ouput_dim=embed_dim, max_position_embeddings=max_road_len, device = device)

        self.road_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
    
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.road_decoder_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, drop_ratio=drop_ratio, attn_drop_ratio=drop_ratio, drop_path_ratio=drop_ratio)#, qk_scale=None
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_embed_road = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, drop_ratio=drop_ratio, attn_drop_ratio=drop_ratio, drop_path_ratio=drop_ratio)#, qk_scale=None
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        
        self.decoder_pred = nn.Linear(decoder_embed_dim, decoder_embed_dim, bias=True)

        self.midlossfc = nn.Linear(decoder_embed_dim, 2, bias=True)
        # --------------------------------------------------------------------------

        self.automodel = AutoPoly(latent_dim = decoder_embed_dim, device = device)

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
  

    def forward_encoder(self, x, pos, road):
        bsz, len_build = x.shape[:2]

        x= F.relu(self.fc_embedding(x.flatten(0,1))).view(bsz, -1, self.embed_dim)
        x_road= F.relu(self.road_embedding(road.flatten(0,1))).view(bsz, -1, self.embed_dim)

        x = x + self.pos_embed_cxy(self.embed_dim, pos)
        x_road = x_road + self.road_token.repeat(bsz, x_road.shape[1], 1)

        x = torch.cat([self.mask_token.repeat(x.shape[0], 1, 1), x, x_road], dim = 1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if self.append:
            return x[:, :1+len_build], x[:, 1+len_build:]
        else:
            return x[:, :1+len_build]

    def forward_decoder(self, x, posall, latent =  None):
        x = F.relu(self.decoder_embed(x))
        if self.append:
            x_road = F.relu(self.decoder_embed_road(latent))

        mask_tokens = x[:, 0:1, :]

        x_ = x[:, 1:, :]

        mask_tokens = mask_tokens.repeat(1, self.max_build-x_.shape[1], 1)
        x_ = torch.cat([x_, mask_tokens], dim = 1)   

        x_ = x_ + self.pos_embed_cxy(self.decoder_embed_dim, posall)

        if self.append:
            len_tem = x_.shape[1]
            x_road = x_road + self.road_decoder_token.repeat(x_road.shape[0], x_road.shape[1], 1)
            x_ = torch.cat([x_, x_road], dim = 1)

        for blk in self.decoder_blocks:
            x_ = blk(x_)

        x_ = self.decoder_norm(x_)
        if self.append:
            out = self.decoder_pred(x_[:, :len_tem])
        else:
            out = self.decoder_pred(x_)

        return out

    def compute_loss(self, out, polyin, len_tar):
        hyp_bsz = out.shape[0]
        
        poly_out = out[:, :, :2]
        poly_len = out[:, :, 2]
        loss_l1 = self.mseloss(torch.cat([poly_out[i, :len_tar[i]] for i in range(hyp_bsz)], dim = 0),
                               torch.cat([polyin[i, :len_tar[i]] for i in range(hyp_bsz)], dim = 0))

        poly_len_tar = torch.zeros(poly_len.shape).scatter_(1, len_tar.unsqueeze(-1), torch.ones(len(len_tar), 1)).to(self.device)
        loss_len = self.bceloss(poly_len, poly_len_tar)

        return loss_l1, loss_len
    

    def forward(self, poly, pos, postar, polytar, len_tar, road):
        bsz, remain_num, _, _ = poly.shape
        posall = torch.cat([pos, postar], dim = 1) 

        if self.append:
            latent, latent_road = self.forward_encoder(poly, pos, road)
            pred_latent = self.forward_decoder(latent, posall, latent_road) 
        else:
            latent = self.forward_encoder(poly, pos, road)
            pred_latent = self.forward_decoder(latent, posall) 

        latentautoin = torch.cat([pred_latent[i, remain_num:remain_num+len(len_tar[i])] for i in range(bsz)], dim = 0)
        polyautoin = torch.cat([polytar[i, :len(len_tar[i])] for i in range(bsz)], dim = 0)

        out = self.automodel(latentautoin, polyautoin)

        len_tar = torch.cat([len_tar[i] for i in range(bsz)], dim = 0)
        loss_l1, loss_len = self.compute_loss(out, polyautoin, len_tar) 

        return loss_l1, loss_len, out

    def infgen(self, poly, pos, postar, road):
        bsz, remain_num, _, _ = poly.shape
        assert bsz == 1
        postar_len = postar.shape[1]
        postarin = torch.zeros([1, self.max_build-remain_num, 2]).to(self.device)
        postarin[:, :postar_len, :] = postar
        posall = torch.cat([pos, postarin], dim = 1)
        assert posall.shape[1]==self.max_build

        if self.append:
            latent, latent_road = self.forward_encoder(poly, pos, road)
            pred_latent = self.forward_decoder(latent, posall, latent_road) 
        else:
            latent = self.forward_encoder(poly, pos, road)
            pred_latent = self.forward_decoder(latent, posall) 

        latentin = pred_latent[0, remain_num:remain_num+postar_len]
        output_poly = []
        for k in range(latentin.shape[0]): 
            latentin_ = latentin[k:k+1]
            out = self.automodel(latentin_, gen = True)
            inputpoly = out[0, -1, :2].unsqueeze(0).unsqueeze(0)

            for i in range(19):
                out = self.automodel(latentin_, inputpoly)
                if torch.sigmoid(out[0, -1, 2])>0.1 and inputpoly.shape[1]>2:
                        # print('end:', torch.sigmoid(out[0, -1, 2]))
                    break
                inputpoly = torch.cat([inputpoly, out[0, -1, :2].unsqueeze(0).unsqueeze(0)], dim = 1)
            output_poly.append(inputpoly)

        return output_poly
                
    