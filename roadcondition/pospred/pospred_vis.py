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
sys.path.append('../')

import timm
import time

#assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import util.lr_sched as lr_sched

from pospred_model import MAGECityPosition, MAGECityPosition_Maxpool, MAGECityPosition_Minlen
from criterion import CityPolyCrit

from dataloader import PolyDataset
import scipy.stats as stats

import cv2

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=1, type=int)


    # Dataset parameters
    parser.add_argument('--data_path', default='../results/states/poly_np.npy', type=str,
                        help='dataset path')
    parser.add_argument('--datapos_path', default='../results/states/polypos_np.npy', type=str,
                        help='dataset path')
    parser.add_argument('--datainfo_path', default='../results/states/polyinfo_np.npy', type=str,
                        help='dataset path')

    parser.add_argument('--save_dir', default='./results_vis',
                        help='path where to tensorboard log')
    parser.add_argument('--model_dir', default='./results_vis',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=20, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.add_argument('--activation', default='sigmoid')

    parser.add_argument('--split_ratio', type = float, default=0.8)
    parser.add_argument('--trans_deep', type = int, default=6)
    parser.add_argument('--trans_deep_decoder', type = int, default=3)
    parser.add_argument('--num_heads', type = int, default=8)

    parser.add_argument('--save_freq', type = int, default=50)
    parser.add_argument('--embed_dim', type = int, default=256)
    parser.add_argument('--decoder_embed_dim', type = int, default=32)

    parser.add_argument('--drop_ratio', type = float, default=0.0)

    parser.add_argument('--remain_num', type = int, default=30)

    parser.add_argument('--pos_weight', type = float, default=100)

    parser.add_argument('--add_maxpool', type = bool, default=False)

    parser.set_defaults(pin_mem=True)

    return parser

def random_masking(x, pos, info, remain_num):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        len_keep = remain_num
        poly_reserve = torch.zeros([x.shape[0], remain_num, 20, 2])
        pos_reserve = torch.zeros([x.shape[0], remain_num, 2])

        pos_tar = []

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
            pos_tar.append(pos_tem[ids_tar])

        return poly_reserve, pos_reserve, pos_tar, ids_keep


def main(args):

    #print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    #print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_valid = PolyDataset(args.data_path, args.datapos_path, args.datainfo_path, train=False,split_ratio = args.split_ratio)

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)

    data_loader_valid = torch.utils.data.DataLoader(
        dataset_valid, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    # define the model
    model = MAGECityPosition_Minlen(drop_ratio = args.drop_ratio, activation=args.activation, num_heads=args.num_heads, device = args.device,
                        depth=args.trans_deep, embed_dim=args.embed_dim, decoder_embed_dim = args.decoder_embed_dim, 
                        discre=args.discre, patch_num=args.patch_num, patch_size=args.patch_size,
                        decoder_depth=args.trans_deep_decoder, decoder_num_heads=args.num_heads, pos_weight = args.pos_weight)
    
    pretrained_model = torch.load(args.model_dir)
    model.load_state_dict(pretrained_model)
    model.to(device)


    remain_num = args.remain_num


    for epoch in range(args.start_epoch, args.epochs):

        for valid_step, (samples, pos, info, road) in enumerate(data_loader_valid):
            if valid_step >=50:
                break
            model.eval()

            remain_num = min(int(torch.randint(0, 2, (1,))), int(torch.min(info[:, 0])))
            poly_reserve, pos_reserve, pos_tar, ids_keep = random_masking(samples, pos, info, remain_num)

            poly_reserve = poly_reserve.to(device)
            pos_reserve = pos_reserve.to(device)    

            img = np.ones((1000,1000,3),np.uint8)*255

            for i in range(int(info[0, 0])):   
                if i not in ids_keep:
                    pts = np.array(samples[0, i][:int(info[0][i+1]), :].cpu(), np.int32)
                    pts = pts.reshape((-1,1,2)).astype(int)
                    cv2.fillPoly(img, [pts], color=(255, 255, 0))
                    cv2.polylines(img,[pts],True,(0,0,0),1)


            for id in ids_keep:
                pts = np.array(samples[0, id][:int(info[0][id+1]), :].cpu(), np.int32)
                pts = pts.reshape((-1,1,2)).astype(int)
                cv2.fillPoly(img, [pts], color=(238, 159, 153))
                cv2.polylines(img,[pts],True,(0,0,0),1)

 
            with torch.no_grad():
                if args.add_maxpool:
                    loss, pred, loss_bce = model(poly_reserve, pos_reserve, pos_tar)
                else:
                    loss, pred = model(poly_reserve, pos_reserve, pos_tar)

            pred = torch.sigmoid(pred).view(100,100,1).permute(1,0,2).repeat(1,1,3).detach().cpu()
            posimg_pred = np.array(pred*255, dtype=np.uint8)
            # index = torch.nonzero(torch.where(pred>0.5, 1.0, 0.0))

            # posimg_pred = np.zeros((50,50,3), dtype=np.uint8)
            # for pos in index:
            #     cv2.circle(posimg_pred,(int(pos[0]), int(pos[1])),0,(255,255,255))
    
            # img = cv2.hconcat([img_t, cv2.resize(posimg_gt,(500,500)), cv2.resize(posimg_pred, (500,500))])
            # img_pred = cv2.hconcat([posimg_gt, posimg_pred])

            cv2.imwrite(f'{args.save_dir}'+str(valid_step)+ 'poly' +'.jpg',img)
            cv2.imwrite(f'{args.save_dir}'+str(valid_step) +'.jpg',posimg_pred)

            print(valid_step, loss.item())      
                  



if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)

    