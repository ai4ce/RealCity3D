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


import timm
import time

#assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import util.lr_sched as lr_sched

from models_mage3d import MAGECityPolyGen3D

from dataloader import PolyDataset3D
import scipy.stats as stats

import cv2

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-6, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='../datasets/3Dpoly/poly_np.npy', type=str,
                        help='dataset path')
    parser.add_argument('--datapos_path', default='../datasets/3Dpoly/polypos_np.npy', type=str,
                        help='dataset path')
    parser.add_argument('--datainfo_path', default='../datasets/3Dpoly/polyinfo_np.npy', type=str,
                        help='dataset path')
    parser.add_argument('--datah_path', default='../datasets/3Dpoly/polyh_np.npy', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='../results/mae/output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='../results/mae/output_log',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')

    parser.add_argument('--split_ratio', type = float, default=0.8)
    parser.add_argument('--trans_deep', type = int, default=12)
    parser.add_argument('--trans_deep_decoder', type = int, default=4)
    parser.add_argument('--num_heads', type = int, default=16)

    parser.add_argument('--save_freq', type = int, default=50)
    parser.add_argument('--embed_dim', type = int, default=32)
    parser.add_argument('--decoder_embed_dim', type = int, default=16)

    parser.add_argument('--drop_ratio', type = float, default=0.1)

    parser.add_argument('--remain_num', type = int, default=6)
    parser.add_argument('--max_poly', type = int, default=20)
    parser.add_argument('--max_build', type = int, default=60)

    parser.add_argument('--loss_weight', type = float, default=100)
    parser.add_argument('--pos_weight', type = float, default=20)

    parser.set_defaults(pin_mem=True)

    return parser

def random_masking(x, pos, info, h, remain_num, max_build = 60):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        len_keep = remain_num
        poly_reserve = torch.zeros([x.shape[0], remain_num, 20, 2])
        pos_reserve = torch.zeros([x.shape[0], remain_num, 2])
        h_reserve = torch.zeros([x.shape[0], remain_num, 1])

        poly_tar = torch.zeros([x.shape[0], max_build-remain_num, 20, 2])
        pos_tar = torch.zeros([x.shape[0], max_build-remain_num, 2])
        h_tar = torch.zeros([x.shape[0], max_build-remain_num, 1])

        len_tar = []
        for i in range(x.shape[0]):
            L = int(info[i, 0])

            x_tem = x[i, :L].clone()
            pos_tem = pos[i, :L].clone()
            h_tem = h[i, :L].clone()

            noise = np.random.rand(L)
            ids_shuffle = np.argsort(noise, axis=0)  
            ids_keep = ids_shuffle[:len_keep]
            ids_tar = ids_shuffle[len_keep:]

            poly_reserve[i] = x_tem[ids_keep]
            pos_reserve[i] = pos_tem[ids_keep]
            h_reserve[i, :, 0] = h_tem[ids_keep]
            poly_tar[i][:len(ids_tar), :, :] = x_tem[ids_tar]
            pos_tar[i][:len(ids_tar), :] = pos_tem[ids_tar]
            h_tar[i][:len(ids_tar), 0] = h_tem[ids_tar]
            len_tar.append(info[i, ids_tar+1].long())

        return poly_reserve, pos_reserve, h_reserve, poly_tar, pos_tar, h_tar, len_tar


def main(args):
    device = torch.device(args.device)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True


    dataset_train = PolyDataset3D(args.data_path, args.datapos_path, args.datainfo_path, args.datah_path, train=True,split_ratio = args.split_ratio)

    dataset_valid = PolyDataset3D(args.data_path, args.datapos_path, args.datainfo_path, args.datah_path,train=False,split_ratio = args.split_ratio)

    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    data_loader_valid = torch.utils.data.DataLoader(
        dataset_valid, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    # define the model
    model = MAGECityPolyGen3D(drop_ratio = args.drop_ratio, num_heads=args.num_heads, device = args.device,
                        depth=args.trans_deep, embed_dim=args.embed_dim, decoder_embed_dim = args.decoder_embed_dim,
                        decoder_depth=args.trans_deep_decoder, decoder_num_heads=args.num_heads, pos_weight = args.pos_weight)
    
    model.to(device)

    model_without_ddp = model

    eff_batch_size = args.batch_size * args.accum_iter 
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    
    remain_num = args.remain_num

    best_valid_loss = 1000000
    train_num = 0
    for epoch in range(args.start_epoch, args.epochs):
        model.train(True)
        optimizer.zero_grad()

        for data_iter_step, (samples, pos, info, h) in enumerate(data_loader_train):

            if data_iter_step % args.accum_iter == 0:
                lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader_train) + epoch, args)
            
            poly_reserve, pos_reserve, h_reserve, poly_tar, pos_tar, h_tar, len_tar = random_masking(samples, pos, info, h, remain_num)

            poly_reserve = poly_reserve.to(device)
            pos_reserve = pos_reserve.to(device)
            h_reserve = h_reserve.to(device)
            pos_tar = pos_tar.to(device)
            poly_tar = poly_tar.to(device)
            h_tar = h_tar.to(device)
            loss_l1, loss_height, loss_len, _ = model(poly_reserve, pos_reserve, h_reserve,  poly_tar, pos_tar, h_tar, len_tar)

            loss = loss_l1 + loss_height + args.loss_weight*loss_len

            loss_value = loss.item()
            
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)
            loss /= args.accum_iter
            if (data_iter_step + 1) % args.accum_iter == 0:
                optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            log_writer.add_scalar('loss_train', loss.item(), train_num)
            log_writer.add_scalar('loss_train_l1', loss_l1.item(), train_num)
            log_writer.add_scalar('loss_train_h', loss_height.item(), train_num)
            log_writer.add_scalar('loss_train_len', loss_len.item(), train_num)

            train_num+=1


        print('train_loss_l1:', loss_l1.item()) 
        print('train_loss_h:', loss_height.item()) 
        print('train_loss_len:', loss_len.item()) 

        valid_loss = 0
        valid_count = 0
        for valid_step, (samples, pos, info, h) in enumerate(data_loader_valid):

            model.eval()

            poly_reserve, pos_reserve, h_reserve, poly_tar, pos_tar, h_tar, len_tar = random_masking(samples, pos, info, h, remain_num)

            poly_reserve = poly_reserve.to(device)
            pos_reserve = pos_reserve.to(device)
            h_reserve = h_reserve.to(device)
            pos_tar = pos_tar.to(device)
            poly_tar = poly_tar.to(device)
            h_tar = h_tar.to(device)
  
            with torch.no_grad():
                loss_l1, loss_height, loss_len, _ = model(poly_reserve, pos_reserve, h_reserve,  poly_tar, pos_tar, h_tar, len_tar)

                loss = loss_l1 + loss_height + args.loss_weight*loss_len

            valid_loss += loss.item()
            valid_count += 1         
                  
        val_loss = valid_loss/valid_count


        print('epoch:', epoch, 'val_loss_l1: ', loss_l1.item())
        print('epoch:', epoch, 'val_loss_h: ', loss_height.item())
        print('epoch:', epoch, 'val_loss_len: ', loss_len.item())


        log_writer.add_scalar('loss_valid', loss.item(), train_num)
        log_writer.add_scalar('loss_valid_l1', loss_l1.item(), train_num)
        log_writer.add_scalar('loss_valid_h', loss_height.item(), train_num)
        log_writer.add_scalar('loss_valid_len', loss_len.item(), train_num)
    

        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            model_path = os.path.join(args.output_dir, f'mae_reconstruction_best.pth')
            torch.save(model.state_dict(), model_path)

        if epoch%args.save_freq == 0:
            model_fpath = os.path.join(args.output_dir, f'mae_reconstruction_{epoch}.pth')
            torch.save(model.state_dict(), model_fpath)

    model_path = os.path.join(args.output_dir, f'mae_reconstruction_final.pth')
    torch.save(model.state_dict(), model_path)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
