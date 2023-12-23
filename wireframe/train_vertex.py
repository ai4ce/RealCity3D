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


#import timm
import time

#assert timm.__version__ == "0.3.2"  # version check
#import timm.optim.optim_factory as optim_factory

from model_vertex import VertexModel
from dataset import Dataset_V

#import cv2

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def get_args_parser():
    parser = argparse.ArgumentParser('PolyGen pre-training', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=200, type=int)

    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='weight decay (default: 0.001)')

    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (absolute lr)')

    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='../datasets/Brooklyn', type=str,
                        help='dataset path')
    
    parser.add_argument('--output_dir', default='../results/train/wireframe/vertex/output_dir_1e-3_modi_dropout',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='../results/train/wireframe/vertex/output_log_1e-3_modi_dropout',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)


    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')

    parser.add_argument('--split_ratio', type = float, default=0.8)

    parser.add_argument('--num-layers', type = int, default=4)

    parser.add_argument('--save_freq', type = int, default=10)
    parser.add_argument('--embedding-dim', type = int, default=256)
    parser.add_argument('--hidden-size', type = int, default=256)

    parser.set_defaults(pin_mem=True)

    return parser


def main(args):
    device = torch.device(args.device)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True


    dataset_train = Dataset_V(args.data_path, train=True, split = args.split_ratio, device = device)

    dataset_valid = Dataset_V(args.data_path, train=False, split = args.split_ratio, device = device)

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
        collate_fn=dataset_train.collate_fn,
        shuffle=True
    )
    data_loader_valid = torch.utils.data.DataLoader(
        dataset_valid, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=dataset_valid.collate_fn,
        shuffle=True
    )
    
    # define the model
    model = VertexModel(args)
    
    model.to(device)
    
    # zhushi below when train, use wen generate
    '''model.load_state_dict(torch.load("models/vmodel30.pth"))
    model.generate([70, 65, 28], "models/v1.obj")
    print("generated models / v1.obj")
    return'''


    model_without_ddp = model
    #param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))

    best_valid_loss = 1000000
    train_num = 0
    tt = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        model.train(True)
        optimizer.zero_grad()

        for data_iter_step, sample in enumerate(data_loader_train):
            #print('xxxxxxxx', data_iter_step)
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader_train) + epoch, args)
            
            source = sample["source"].to(device)
            target = sample["target"].to(device)
            loss = model.get_loss(source, target)

            loss_value = loss.item()
            
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            log_writer.add_scalar('loss_train', loss.item(), train_num)

            train_num+=1


        print('train_loss:', loss.item()) 

        valid_loss = 0
        valid_count = 0
        for valid_step, sample in enumerate(data_loader_valid):

            model.eval()

            source = sample["source"].to(device)
            target = sample["target"].to(device)
            
            with torch.no_grad():
                loss = model.get_loss(source, target)

            valid_loss += loss.item()
            valid_count += 1         
                  
        val_loss = valid_loss/valid_count

        print('epoch:', epoch, 'val_loss: ', val_loss, "time:",time.time()-tt)
        tt = time.time()


        log_writer.add_scalar('loss_valid', val_loss, train_num)

        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            model_path = os.path.join(args.output_dir, f'mae_reconstruction_best.pth')
            torch.save(model.state_dict(), model_path)

        if epoch % args.save_freq == 0:
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
