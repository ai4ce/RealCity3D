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
from functools import partial
import torch.nn.functional as F
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
from scipy.optimize import linear_sum_assignment
import time

from util.pos_embed import get_2d_sincos_pos_embed
# from kfiou.rotated_iou_loss import RotatedIoULoss, rotated_iou_loss
# from mmcv.ops import diff_iou_rotated_2d
from oriented_iou_loss import cal_giou

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, max_step=32, action_space = 6,
                 embed_dim=128, depth=12, num_heads=16,
                 decoder_embed_dim=64, decoder_depth=4, decoder_num_heads=8,
                 mlp_ratio=4.,norm_layer=nn.LayerNorm ,mask_ratio=0.6, 
                 device='cuda', activation='relu', patch_size = 5, max_num_box = 8, pos_weight = 5):
        super().__init__()
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.device = device
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.activation = activation
        self.mask_ratio = mask_ratio
        self.max_step =max_step
        self.action_space = action_space

        self.patch_size = patch_size
        self.num_patches = patch_size**2
        self.patch_len = 1/patch_size
        self.max_num_box = max_num_box
        self.num_heads = num_heads

        if self.activation == 'relu':
            self.activation_fun = F.relu
        elif self.activation == 'sigmoid':
            self.activation_fun = torch.sigmoid

        self.fc_embedding =  nn.Sequential(nn.Linear(self.action_space, embed_dim, bias=True))
      
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim*max_num_box), requires_grad=False) 

        self.blocks = nn.ModuleList([
            Block(embed_dim*max_num_box, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)#, qk_scale=None
            for i in range(depth)])
        self.norm = norm_layer(embed_dim*max_num_box)
        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim*max_num_box))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_embed_dim*max_num_box), requires_grad=False)
 
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim*max_num_box, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)#, qk_scale=None
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim*max_num_box)

        self.decoder_pred = nn.Linear(decoder_embed_dim, self.action_space, bias=True) # decoder to patch

        # --------------------------------------------------------------------------

        self.initialize_weights()
        self.loss_l1 = nn.L1Loss()
        self.class_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

    def initialize_weights(self):
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches**.5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.num_patches**.5), cls_token=False)
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

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        B, P, A = x.shape  
        len_keep = int(P * (1 - mask_ratio))
        
        noise = torch.rand(B, P, device=x.device)  

        ids_shuffle = torch.argsort(noise, dim=1)  
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]

        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, A))

        mask = torch.ones([B, P], device=x.device)
        mask[:, :len_keep] = 0

        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore


    def forward_encoder(self, x, mask_ratio):

        x= F.relu(self.fc_embedding(x))

        x = x.flatten(2,3) 
        x = x + self.pos_embed

        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x_unflatten = x.view(-1, int(self.num_patches * (1 - self.mask_ratio)), self.max_num_box, self.embed_dim)

        x = self.decoder_embed(x_unflatten)

        x = x.flatten(2,3)

        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1],1)

        x_ = torch.cat([x, mask_tokens], dim=1) 

        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_.shape[2]))  # unshuffle
              
        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)

        x = self.decoder_norm(x)

        out = x.view(-1, self.num_patches, self.max_num_box, self.decoder_embed_dim)

        out = self.decoder_pred(out)

        out =  torch.cat([self.activation_fun(out[:,:,:,:4]), out[:,:,:,4:]],dim=-1)
        
        return out
    
    def box2corners_th(self, box):
        """convert box coordinate to corners

        Args:
            box (torch.Tensor): (B, N, 5) with x, y, w, h, alpha

        Returns:
            torch.Tensor: (B, N, 4, 2) corners
        """
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

    def cal_box_l1(self, output_box, target_box):
        return (torch.abs(output_box[:,:,:4]-target_box[:,:,:4]).sum(-1) + 
                torch.abs(torch.sin(output_box[:,:,4])-torch.sin(target_box[:,:,4])) + 
                torch.abs(torch.cos(output_box[:,:,4])-torch.cos(target_box[:,:,4])))

    def matcher_batch(self, outputs, targets):

        outputs = outputs.flatten(0, 1)
        targets = targets.flatten(0, 1)
        bs, num_box, action_space = outputs.shape

        ####
        output_box_l = outputs.permute(1,0,2)[:, :, :5]
        target_len = torch.count_nonzero(targets[:,:,5], dim = 1)
        # print('target_len:', target_len)
        target_box_l = torch.cat([torch.gather(targets[i], index=targets[i][:,5].nonzero().repeat(1, action_space), dim = 0) 
                                  for i in range(targets.shape[0])], dim = 0)[:, :5]
        
        output_class_l = outputs.permute(1,0,2)[:, :, 5]
        cost_class = torch.repeat_interleave(output_class_l, target_len, 1)  

        cost_bbox, _ = cal_giou(torch.repeat_interleave(output_box_l, target_len, 1), target_box_l.unsqueeze(0).repeat(num_box, 1, 1), 'pca')

        cost_l1 = torch.abs(self.box2corners_th(torch.repeat_interleave(output_box_l, target_len, 1)) - 
                            self.box2corners_th(target_box_l.unsqueeze(0).repeat(num_box, 1, 1))).sum(-1).sum(-1)
        
        cost_box_l1 = self.cal_box_l1(torch.repeat_interleave(output_box_l, target_len, 1), target_box_l.unsqueeze(0).repeat(num_box, 1, 1))
        C = (cost_bbox - torch.sigmoid(cost_class) + cost_l1 + cost_box_l1).detach().cpu() 
        
        indices = [linear_sum_assignment(c) for c in C.split(target_len.detach().cpu().numpy().tolist(), -1)]
        ####

        ####
        output_class = outputs[:, :, 5]
        target_class = torch.cat([torch.zeros(num_box).scatter_(0, torch.as_tensor(indices[i][0], dtype=torch.int64), 
                                                                torch.ones(target_len[i])).unsqueeze(0) for i in range(bs)], dim = 0).to(output_class.device)

        loss_class = self.class_criterion(output_class, target_class)
        ####

        ####
        output_box = torch.cat([torch.gather(outputs[i], index = torch.as_tensor(indices[i][0], dtype=torch.int64)
                                             .unsqueeze(-1).repeat(1, action_space-1).to(outputs.device), dim=0) for i in range(bs)], dim = 0)
        target_box = torch.cat([torch.gather(torch.gather(targets[i], index=targets[i][:,5].nonzero().repeat(1, action_space), dim = 0), 
                                             index = torch.as_tensor(indices[i][1], dtype=torch.int64)
                                             .unsqueeze(-1).repeat(1, action_space-1).to(targets.device), dim=0) for i in range(bs)], dim = 0)

        loss_giou, _ = cal_giou(output_box.unsqueeze(0), target_box.unsqueeze(0), 'pca')
        #loss_giou = self.giou(output_box.unsqueeze(0), target_box.unsqueeze(0))
        # print('loss_giou: ',loss_giou)
        loss_giou = loss_giou.sum()/(loss_giou.shape[0]*loss_giou.shape[1])
        ####

        ####
        loss_l1 = torch.abs(self.box2corners_th(output_box.unsqueeze(0))-self.box2corners_th(target_box.unsqueeze(0)))

        loss_l1 = loss_l1.sum()/(loss_l1.shape[0]*loss_l1.shape[1])
        ####

        ####
        loss_box_l1 = self.cal_box_l1(output_box.unsqueeze(0),target_box.unsqueeze(0))

        loss_box_l1 = loss_box_l1.sum()/(loss_box_l1.shape[0]*loss_box_l1.shape[1])
        ####
        

        loss = loss_giou + loss_class + loss_l1 + loss_box_l1

        return loss, loss_giou, loss_class, loss_l1

     
    def forward(self, imgs, mask_ratio):

        x_pad = imgs.float()
        latent, mask, ids_restore = self.forward_encoder(x_pad, mask_ratio)

        pred = self.forward_decoder(latent, ids_restore) 

        mask = mask.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, pred.shape[2], pred.shape[3]).bool()

        pred1 = torch.masked_select(pred, mask)

        pred = pred1.view(-1, self.num_patches-int(self.num_patches*(1-self.mask_ratio)), pred.shape[2], pred.shape[3])
        x_pad = torch.masked_select(x_pad, mask).view(-1, self.num_patches-int(self.num_patches*(1-self.mask_ratio)), pred.shape[2], pred.shape[3])


        loss, loss_giou, loss_class, loss_l1 = self.matcher_batch(pred, x_pad)
        return loss, pred, mask, loss_giou, loss_class, loss_l1
        
