import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
from util.transformer import Block
from reformer_pytorch import Reformer
import time
import sys

from util.pos_embed import get_2d_sincos_pos_embed

import scipy.stats as stats

class MAGECityGenPospred(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, max_step=32, action_space = 5, mask_ratio_min = 0.5,
                 embed_dim=256, depth=12, num_heads=8, mask_ratio_max=1.0, mask_ratio_mu=0.55, mask_ratio_std=0.25,
                 decoder_embed_dim=64, decoder_depth=4, decoder_num_heads=8, pos_weight = 100,
                 mlp_ratio=4., drop_ratio = 0.1, discre = 50, patch_num = 10, patch_size = 5,
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
        self.patch_num = patch_num
        self.patch_size = patch_size

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

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.patch_num**2, decoder_embed_dim*(self.patch_size**2)), requires_grad=False)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim*(self.patch_size**2), decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, drop_ratio=drop_ratio, attn_drop_ratio=drop_ratio, drop_path_ratio=drop_ratio)#, qk_scale=None
            for i in range(decoder_depth)])
        

        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.decoder_pred = nn.Linear(decoder_embed_dim, 1) # decoder to patch


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


    def forward_encoder(self, x, pos_embed):
        x= F.relu(self.fc_embedding(x))

        x = x + self.pos_embed_cxy(self.embed_dim, pos_embed)

        x = torch.cat([self.mask_token.repeat(x.shape[0], 1, 1), x], dim = 1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward_decoder(self, x, pos):
        x = self.decoder_embed(x)

        mask_tokens = x[:, 0, :].unsqueeze(1)
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

    def forward(self, imgs, mask_ratio):
        bsz, _, _ = imgs.shape

        mask_ratio = self.mask_ratio_generator.rvs(1)[0]

        imgs_fea = imgs[:, :, :5]
        position = imgs[:, :, 5:]

        x_reserve, mask, pos_reserve, x_tar, pos_tar = self.random_masking(imgs_fea, position, mask_ratio)
        latent = self.forward_encoder(x_reserve, pos_reserve)

        pred = self.forward_decoder(latent, pos_reserve).squeeze(-1)  

        tar_pos = torch.cat([torch.zeros(self.discre**2, device=self.device).scatter_(0, (pos_tar[b][:, 0]*self.discre+pos_tar[b][:, 1]).long().to(self.device), 
                                                                                      torch.ones(pos_tar[b].shape[0], device=pred.device)).unsqueeze(0) for b in range(bsz)], dim = 0)

        loss = self.bceloss(pred, tar_pos)

        return loss, pred, mask




    