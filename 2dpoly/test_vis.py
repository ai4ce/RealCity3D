# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import math
import sys
import os
import time
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm
import time

#assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import util.lr_sched as lr_sched

from models_mage import MAGECityPolyGen

from dataloader import PolyDataset
import scipy.stats as stats

import cv2

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Dataset parameters
    parser.add_argument('--data_path', default='../results/states/poly_np.npy', type=str,
                        help='dataset path')
    parser.add_argument('--datapos_path', default='../results/states/polypos_np.npy', type=str,
                        help='dataset path')
    parser.add_argument('--datainfo_path', default='../results/states/polyinfo_np.npy', type=str,
                        help='dataset path')
    
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')

    parser.add_argument('--split_ratio', type = float, default=0.8)
    parser.add_argument('--trans_deep', type = int, default=12)
    parser.add_argument('--trans_deep_decoder', type = int, default=8)
    parser.add_argument('--num_heads', type = int, default=8)

    parser.add_argument('--save_freq', type = int, default=50)
    parser.add_argument('--embed_dim', type = int, default=512)
    parser.add_argument('--decoder_embed_dim', type = int, default=256)

    parser.add_argument('--drop_ratio', type = float, default=0.0)
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--draw_dir', default='../results/statepoly_draw_auto/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--model_path', default='/home/rl4citygen/DRL4CityGen/results/polyauto/output_dir/64_800_12_8_8_512_256_0.0_pos20_100/mae_reconstruction_best.pth')

    return parser

def random_masking(x, pos, info, remain_num, max_build = 60):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        len_keep = remain_num
        poly_reserve = torch.zeros([x.shape[0], remain_num, 20, 2])
        pos_reserve = torch.zeros([x.shape[0], remain_num, 2])

        poly_tar = torch.zeros([x.shape[0], max_build-remain_num, 20, 2])
        pos_tar = torch.zeros([x.shape[0], max_build-remain_num, 2])

        len_tar = []

        for i in range(x.shape[0]):
            L = int(info[i, 0])

            x_tem = x[i, :L].clone()
            pos_tem = pos[i, :L].clone()

            noise = np.random.rand(L)
            ids_shuffle = np.argsort(noise, axis=0)  
            ids_keep = ids_shuffle[:len_keep]
            ids_tar = ids_shuffle[len_keep:]

            poly_reserve[i] = x_tem[ids_keep]
            pos_reserve[i] = pos_tem[ids_keep]
            poly_tar[i][:len(ids_tar), :, :] = x_tem[ids_tar]
            pos_tar[i][:len(ids_tar), :] = pos_tem[ids_tar]
            len_tar.append(info[i, ids_tar+1].long())

        return poly_reserve, pos_reserve, poly_tar, pos_tar, len_tar, ids_keep


def main(args):

    #print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    #print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_valid = PolyDataset(args.data_path, args.datapos_path, args.datainfo_path, train=False,split_ratio = args.split_ratio)

    data_loader_valid = torch.utils.data.DataLoader(
        dataset_valid, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    # define the model
    model = MAGECityPolyGen(drop_ratio = args.drop_ratio, num_heads=args.num_heads, device = args.device,
                        depth=args.trans_deep, embed_dim=args.embed_dim, decoder_embed_dim = args.decoder_embed_dim,
                        decoder_depth=args.trans_deep_decoder, decoder_num_heads=args.num_heads)
    
    pretrained_model = torch.load(args.model_path)
    model.load_state_dict(pretrained_model)
    model.to(device)


    max_poly = 20
    max_bulid = 60


    model.eval()

    image_num = 0

    for epoch in range(args.epochs):
        for valid_step, (samples, pos, info) in enumerate(data_loader_valid):
            
            assert samples.shape[0] == 1

            remain_num = 6
            poly_reserve, pos_reserve, poly_tar, pos_tar, len_tar, ids_keep = random_masking(samples, pos, info, remain_num)

            poly_reserve = poly_reserve.to(device)
            pos_reserve = pos_reserve.to(device)
            poly_tar = poly_tar.to(device)
            pos_tar = pos_tar.to(device)

            # loss_l1, loss_len, out  = model(poly_reserve, pos_reserve, pos_tar, poly_tar, len_tar)
            # print('loss:', loss_l1, loss_len)
               
            img_t = np.zeros((500,500,3),np.uint8)
            img_p = np.zeros((500,500,3),np.uint8)

            flag = 0
            for i in range(int(info[0, 0])):   
                pts = np.array(samples[0, i][:int(info[0][i+1]), :].cpu(), np.int32)
                pts = pts.reshape((-1,1,2)).astype(int)
                cv2.polylines(img_t,[pts],True,(0,255,255),1)
                # if i not in ids_keep:
                #     if flag == 1:
                #         print(pts)
                #     flag += 1

            for id in ids_keep:
                pts = np.array(samples[0, id][:int(info[0][id+1]), :].cpu(), np.int32)
                pts = pts.reshape((-1,1,2)).astype(int)
                cv2.polylines(img_p,[pts],True,(0,255,255),1)

            
            # for i in range(out.shape[0]):
            #     pol = out[i,:, :2]
            #     pl = out[i, :, 2]
            #     for j in range(out.shape[1]):
            #         if pl[j]>0.5:
            #             idx = j
            #     print(idx)
            #     pts = np.array(pol[:idx, :].detach().cpu(), np.int32)
            #     pts = pts.reshape((-1,1,2)).astype(int)
            #     cv2.polylines(img_p,[pts],True,(255,0,255),1)
            #     # if i == 0:
            #     #     print(pts)
                        

            with torch.no_grad():        
                img_p  = model.generate(poly_reserve, pos_reserve, pos_tar, len_tar, img_p)

            img = cv2.hconcat([img_t, img_p])
                
            dir_path = args.draw_dir
            if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
            image_num += 1
            print('img_num:', image_num)
                        
            cv2.imwrite(f'{dir_path}'+str(image_num) +'.jpg',img)
  


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.draw_dir:
        Path(args.draw_dir).mkdir(parents=True, exist_ok=True)
    main(args)