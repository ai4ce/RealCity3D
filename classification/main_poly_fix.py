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

import util.lr_sched as lr_sched

from models_mage_class import MAGECityPolyGen

from dataloader import PolyDataset

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
    parser.add_argument('--data_path', default='../results/states/poly_np.npy', type=str,
                        help='dataset path')
    parser.add_argument('--datapos_path', default='../results/states/polypos_np.npy', type=str,
                        help='dataset path')
    parser.add_argument('--datainfo_path', default='../results/states/polyinfo_np.npy', type=str,
                        help='dataset path')
    parser.add_argument('--dataclass_path', default='../results/states/polyclass_np.npy', type=str,
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
    parser.add_argument('--num_heads', type = int, default=16)

    parser.add_argument('--save_freq', type = int, default=50)
    parser.add_argument('--embed_dim', type = int, default=32)

    parser.add_argument('--drop_ratio', type = float, default=0.1)

    parser.add_argument('--max_poly', type = int, default=20)
    parser.add_argument('--max_build', type = int, default=60)

    parser.set_defaults(pin_mem=True)

    return parser


def main(args):


    device = torch.device(args.device)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True


    dataset_train = PolyDataset(args.data_path, args.datapos_path, args.datainfo_path, args.dataclass_path, train=True,split_ratio = args.split_ratio)

    dataset_valid = PolyDataset(args.data_path, args.datapos_path, args.datainfo_path, args.dataclass_path, train=False,split_ratio = args.split_ratio)

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
    model = MAGECityPolyGen(drop_ratio = args.drop_ratio, num_heads=args.num_heads, device = args.device,
                        depth=args.trans_deep, embed_dim=args.embed_dim)
    
    model.to(device)

    model_without_ddp = model

    eff_batch_size = args.batch_size * args.accum_iter 
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    

    best_valid_loss = 1000000
    train_num = 0
    for epoch in range(args.start_epoch, args.epochs):
        model.train(True)
        optimizer.zero_grad()

        for data_iter_step, (samples, pos, info, class_name) in enumerate(data_loader_train):

            if data_iter_step % args.accum_iter == 0: 
                lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader_train) + epoch, args)

            poly = samples.to(device).float()
            pos = pos.to(device).float()
            class_name = class_name.to(device).long()
            loss, acc = model(poly, pos, class_name)

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
            log_writer.add_scalar('acc_train', acc.item(), train_num)

            train_num+=1


        print('train_loss:', loss.item()) 
        print('train_acc:', acc.item()) 

        valid_loss = 0
        val_acc = 0
        valid_count = 0
        for valid_step, (samples, pos, info, class_name) in enumerate(data_loader_valid):

            model.eval()

            poly = samples.to(device).float()
            pos = pos.to(device).float()
            class_name = class_name.to(device).long()
            
               
            with torch.no_grad():
                loss, acc = model(poly, pos, class_name)

            valid_loss += loss.item()
            valid_count += 1 
            val_acc += acc.item()    
                  
        val_loss = valid_loss/valid_count
        val_acc = val_acc/valid_count

        print('epoch:', epoch, 'val_loss: ', val_loss)
        print('epoch:', epoch, 'val_acc: ', val_acc)

        log_writer.add_scalar('loss_valid', val_loss, train_num)
        log_writer.add_scalar('acc_valid', val_acc, train_num)

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
