import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
from util.transformer import Block
from reformer_pytorch import Reformer
import time
import sys

sys.path.append('../')
from util.pos_embed import get_2d_sincos_pos_embed
from poly_embed import PolyEmbed
from models_autoregress import AutoPoly

import cv2

class MAGECityPosition(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, embed_dim=256, depth=12, num_heads=8,   
                 decoder_embed_dim=64, decoder_depth=4, decoder_num_heads=8, pos_weight = 100,
                 mlp_ratio=4., drop_ratio = 0.1,  discre = 50, device = 'cuda',
                 norm_layer=nn.LayerNorm):
        super().__init__()
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.device = device
        self.discre = discre

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

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.discre**2, decoder_embed_dim), requires_grad=False)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, drop_ratio=drop_ratio, attn_drop_ratio=drop_ratio, drop_path_ratio=drop_ratio)#, qk_scale=None
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        
        self.decoder_pred = nn.Linear(decoder_embed_dim, 1)
        
        # --------------------------------------------------------------------------

        self.bceloss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

        self.initialize_weights()

    def initialize_weights(self):
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.discre), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

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

    def forward_decoder(self, x, pos):
        x = self.decoder_embed(x)

        mask_tokens = x[:, 0:1, :]
        out = x[:, 1:, :]

        mask_tokens = mask_tokens.repeat(1, self.discre**2, 1)
        pos_sort = (pos[:,:,0]*self.discre + pos[:,:,1]).unsqueeze(-1).repeat(1,1,out.shape[-1]).long()
        out = mask_tokens.scatter_(1, pos_sort, out)       

        out = out + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            out = blk(out)

        out = self.decoder_norm(out)
        out = self.decoder_pred(out)

        return out

    
    def forward(self, poly, pos, h, postar):
        ## len_tar object_numpy [[]]
        ## poly_tar [x.shape[0], max_build-remain_num, 20, 2]
        bsz, _, _, _ = poly.shape

        latent = self.forward_encoder(poly, pos, h)

        pred = self.forward_decoder(latent, pos).squeeze(-1)

        tar_pos = torch.cat([torch.zeros(self.discre**2, device=self.device).scatter_(0, (postar[b][:, 0]*self.discre+postar[b][:, 1]).long().to(self.device), 
                                                                                      torch.ones(postar[b].shape[0], device=pred.device)).unsqueeze(0) for b in range(bsz)], dim = 0)

        loss = self.bceloss(pred, tar_pos)

        return loss, pred



class MAGECityPosition_Minlen(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, embed_dim=256, depth=12, num_heads=8,  patch_size = 5, patch_num = 10, 
                 decoder_embed_dim=64, decoder_depth=4, decoder_num_heads=8, pos_weight = 100,
                 mlp_ratio=4., drop_ratio = 0.1,  discre = 50, device = 'cuda',
                 norm_layer=nn.LayerNorm):
        super().__init__()
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.device = device
        self.discre = discre
        self.patch_size = patch_size
        self.patch_num = patch_num

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

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.patch_num**2, decoder_embed_dim*(self.patch_size**2)), requires_grad=False)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim*(self.patch_size**2), decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, drop_ratio=drop_ratio, attn_drop_ratio=drop_ratio, drop_path_ratio=drop_ratio)#, qk_scale=None
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        
        self.decoder_pred = nn.Linear(decoder_embed_dim, 1)
        
        # --------------------------------------------------------------------------

        self.bceloss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        self.patchify_func = torch.nn.Unfold((self.patch_size, self.patch_size), stride=self.patch_size)
        self.unpatchify_func = torch.nn.Fold(output_size=(self.discre, self.discre), kernel_size = (self.patch_size, self.patch_size), stride=self.patch_size)

        self.initialize_weights()

    def initialize_weights(self):
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_num), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

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
    

    def forward_decoder(self, x, pos):
        x = self.decoder_embed(x)

        mask_tokens = x[:, 0:1, :]
        out = x[:, 1:, :]

        mask_tokens = mask_tokens.repeat(1, self.discre**2, 1)
        pos_sort = (pos[:,:,0]*self.discre + pos[:,:,1]).unsqueeze(-1).repeat(1,1,out.shape[-1]).long()
        out = mask_tokens.scatter_(1, pos_sort, out)  

        out = out.view(-1, self.discre, self.discre,  self.decoder_embed_dim) 
        out = self.patchify_func(out.permute(0,3,1,2)).permute(0,2,1)    

        out = out + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            out = blk(out)

        out = self.unpatchify_func(out.permute(0,2,1)).flatten(-2, -1).permute(0,2,1)
        out = self.decoder_norm(out)
        out = self.decoder_pred(out)

        return out

    
    def forward(self, poly, pos, h, postar, generate = False):
        bsz, _, _, _ = poly.shape

        latent = self.forward_encoder(poly, pos, h)

        pred = self.forward_decoder(latent, pos).squeeze(-1)
        if generate:
            return pred

        tar_pos = torch.cat([torch.zeros(self.discre**2, device=self.device).scatter_(0, (postar[b][:, 0]*self.discre+postar[b][:, 1]).long().to(self.device), 
                                                                                      torch.ones(postar[b].shape[0], device=pred.device)).unsqueeze(0) for b in range(bsz)], dim = 0)

        loss = self.bceloss(pred, tar_pos)
        
        return loss, pred


class MAGECityPosition_Maxpool(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, embed_dim=256, depth=12, num_heads=8,  patch_size = 5, patch_num = 10, 
                 decoder_embed_dim=64, decoder_depth=4, decoder_num_heads=8, pos_weight = 100,
                 mlp_ratio=4., drop_ratio = 0.1,  discre = 50, device = 'cuda',
                 norm_layer=nn.LayerNorm):
        super().__init__()
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.device = device
        self.discre = discre
        self.patch_size = patch_size
        self.patch_num = patch_num

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

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.patch_num**2, decoder_embed_dim*(self.patch_size**2)), requires_grad=False)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim*(self.patch_size**2), decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, drop_ratio=drop_ratio, attn_drop_ratio=drop_ratio, drop_path_ratio=drop_ratio)#, qk_scale=None
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        
        self.decoder_pred = nn.Linear(decoder_embed_dim, 1)
        
        # --------------------------------------------------------------------------

        self.bceloss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        self.bceloss1 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(20))
        self.bceloss2 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2))

        self.patchify_func = torch.nn.Unfold((self.patch_size, self.patch_size), stride=self.patch_size)
        self.unpatchify_func = torch.nn.Fold(output_size=(self.discre, self.discre), kernel_size = (self.patch_size, self.patch_size), stride=self.patch_size)


        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=5)
        self.initialize_weights()

    def initialize_weights(self):
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_num), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

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
    

    def forward_decoder(self, x, pos):
        x = self.decoder_embed(x)

        mask_tokens = x[:, 0:1, :]
        out = x[:, 1:, :]

        mask_tokens = mask_tokens.repeat(1, self.discre**2, 1)
        pos_sort = (pos[:,:,0]*self.discre + pos[:,:,1]).unsqueeze(-1).repeat(1,1,out.shape[-1]).long()
        out = mask_tokens.scatter_(1, pos_sort, out)  

        out = out.view(-1, self.discre, self.discre,  self.decoder_embed_dim) 
        out = self.patchify_func(out.permute(0,3,1,2)).permute(0,2,1)    

        out = out + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            out = blk(out)

        out = self.unpatchify_func(out.permute(0,2,1)).flatten(-2, -1).permute(0,2,1)
        out = self.decoder_norm(out)
        out = self.decoder_pred(out)

        return out

    
    def forward(self, poly, pos, h, postar):
        bsz, _, _, _ = poly.shape

        latent = self.forward_encoder(poly, pos, h)

        pred = self.forward_decoder(latent, pos).squeeze(-1).reshape(bsz, self.discre, self.discre)

        tar_pos = torch.cat([torch.zeros(self.discre**2, device=self.device).scatter_(0, (postar[b][:, 0]*self.discre+postar[b][:, 1]).long().to(self.device), 
                                                                                      torch.ones(postar[b].shape[0], device=pred.device)).unsqueeze(0) for b in range(bsz)], dim = 0).reshape(bsz, self.discre, self.discre)

        # loss = self.bceloss(pred, tar_pos)

        loss_1 = self.bceloss(pred, tar_pos)
        
        pred_2 = self.maxpool1(pred)
        tar_pos_2 = self.maxpool1(tar_pos)
        loss_2 = self.bceloss1(pred_2, tar_pos_2)

        pred_3 = self.maxpool2(pred_2)
        tar_pos_3 = self.maxpool2(tar_pos_2)
        loss_3 = self.bceloss2(pred_3, tar_pos_3)

        loss = loss_1+loss_2+loss_3
        
        return loss, pred.flatten(-2,-1), loss_1



