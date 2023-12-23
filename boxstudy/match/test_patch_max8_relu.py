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

#assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import util.lr_sched as lr_sched

from models_mae_patch_drop import MaskedAutoencoderViT

from engine_pretrain import train_one_epoch

# import dataloader as maedataset
from dataloader import ValidDataset
import math
import numpy as np
from collections import defaultdict

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import os
import cv2

import sys
import torch.nn.functional as F
def draw_polygon_c(img,point, txt, center, color):
    #print('point: ',point)  # draw box on canvas
    
    cv2.line(img, tuple(point[1]), tuple(point[0]), color, 1)
    cv2.line(img, tuple(point[1]), tuple(point[2]), color, 1)
    cv2.line(img, tuple(point[2]), tuple(point[3]), color, 1)
    cv2.line(img, tuple(point[0]), tuple(point[3]), color, 1)
    return img
def rotate_xy(p, sin, cos, center):
    x_ = (p[:,0:1]-center[:,0:1])*cos-(p[:,1:2]-center[:,1:2])*sin+center[:,0:1]
    y_ = (p[:,0:1]-center[:,0:1])*sin+(p[:,1:2]-center[:,1:2])*cos+center[:,1:2]
#     print(((p[:,0:1]-center[:,0:1])*cos).shape, cos.shape, x_.shape)
    return np.hstack((x_, y_))
def state2img(box,color):
        '''
        parameter:
        box:[number_of _boxes, 5(center_x,center_y, w, h, angle)]
        '''
        X_tree0 = box[:,:2]
        #print('X_tree0.shape',X_tree0.shape)
        Feature_tree0 = box[:, 2:]
        alf = Feature_tree0[:,2:]

        cosO = np.cos(alf)
        sinO = np.sin(alf)

        # sinO = np.cos(Feature_tree0[:,2:3])
        # cosO = np.cos(Feature_tree0[:,2:3])
        #print(cosO)
        #sinO = np.sin(np.arccos(Feature_tree0[:,2:3])) 
        ld = np.hstack((X_tree0[:,0:1]-Feature_tree0[:,0:1]/2, X_tree0[:,1:2]-Feature_tree0[:,1:2]/2))
        rd = np.hstack((X_tree0[:,0:1]+Feature_tree0[:,0:1]/2, X_tree0[:,1:2]-Feature_tree0[:,1:2]/2))
        ru = np.hstack((X_tree0[:,0:1]+Feature_tree0[:,0:1]/2, X_tree0[:,1:2]+Feature_tree0[:,1:2]/2))
        lu = np.hstack((X_tree0[:,0:1]-Feature_tree0[:,0:1]/2, X_tree0[:,1:2]+Feature_tree0[:,1:2]/2))
        ld_r = rotate_xy(ld, sinO, cosO, X_tree0)
        rd_r = rotate_xy(rd, sinO, cosO, X_tree0)
        ru_r = rotate_xy(ru, sinO, cosO, X_tree0)
        lu_r = rotate_xy(lu, sinO, cosO, X_tree0)
        box_r = np.hstack((ld_r*300, rd_r*300, ru_r*300, lu_r*300)).reshape(len(X_tree0), -1, 2)
        #img = np.zeros([300,300],dtype=np.int16)
        img = np.ones(shape=(400,400,3), dtype=np.int16)
        for j, p in enumerate(box_r):
            # if p[0][0]>=0:
            p = np.array(p).astype(int)
            #     #print(p)
            img = draw_polygon_c(img,p, j, X_tree0[j,:2],color)
        return img
def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    parser.add_argument('--mask_ratio', default=0.6, type=float,
                        help='Masking ratio (percentage of removed patches).')

    # Dataset parameters
    parser.add_argument('--data_path', default='../../datasets/boxstates/states_patch.npy', type=str,
                        help='dataset path')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.add_argument('--activation', default='sigmoid')
    parser.add_argument('--split_ratio', type = float, default=0.8)
    parser.add_argument('--trans_deep', type = int, default=4)
    parser.add_argument('--num_heads', type = int, default=8)
    parser.add_argument('--no_pos', type = bool, default=False)
    parser.add_argument('--if_angle', type = bool, default=True)
    parser.add_argument('--model_path', type = str, default='/home/rl4citygen/DRL4CityGen/results/maepatch_chamfer/output_dir/128_1e-3_32_4_rand/mae_reconstruction_final.pth')
    parser.add_argument('--save_path', type = str, default="../../results/test/box/states_draw_patch/")
    parser.add_argument('--patch', type = int, default=5)
    parser.add_argument('--save_freq', type = int, default=50)
    parser.add_argument('--embed_dim', type = int, default=256)
    parser.add_argument('--decoder_embed_dim', type = int, default=64)
    parser.add_argument('--drop_ratio', type = float, default=0.1)
    parser.set_defaults(pin_mem=True)

    return parser

def getboxes_frompred(pred,mask):
        '''
        parameters:
        pred:
        '''
        mask = mask.sum(-1).sum(-1)/48
        box = pred[:,:,:,:5]
        label = pred[:,:,:,5]
        # print('pred.shape: ',pred.shape)
        # print('mask.shape: ',mask.shape)
        print('box.shape: ',box.shape)
        # print('label.shape: ',label.shape)
        '''
        samples.shape:  torch.Size([8, 25, 8, 6])
        pred.shape:  torch.Size([8, 25, 8, 6])
        mask.shape:  torch.Size([8, 25])
        box.shape:  torch.Size([8, 25, 8, 5])
        label.shape:  torch.Size([8, 25, 8])
        '''
        label = torch.sigmoid(label)

        label = torch.where(label <= 0.5, 0.0, 1.0)
        label_index = label.unsqueeze(-1).repeat(1,1,1,5)
        #print('label_index.shape: ',label_index.shape)
        box = box*label_index
        boxes = []
        box = box.flatten(0,1)
        #print('box.shape: ',box.shape)
        box_num = 0
        for i in range(mask.shape[0]):
            temp = []
            for j in range(mask.shape[1]):
                if mask[i,j] == 1:
                    temp.append(box[box_num])
                    box_num += 1
                else:
                    temp.append(torch.zeros((8,5)).float())
            boxes.append(torch.tensor([item.cpu().detach().numpy() for item in temp]).cuda())
        boxes = torch.tensor([item.cpu().detach().numpy() for item in boxes]).cuda()
        #print('boxes.shape: ',boxes.shape)
        #boxes = boxes.view(8, 25, 8, 5)

        center=[]
        center_x = 0
        center_y = 0
        length_patch = 1/args.patch
        for i in range(args.patch**2):
            center.append(torch.tensor([center_x, center_y]))
            if (i+1) %args.patch == 0:
                center_y+=length_patch
                center_x = 0
            else:
                center_x+=length_patch

        for i in range(boxes.shape[0]):
            for j in range(boxes.shape[1]):
                node_x,node_y = center[j]
                #print('boxes[i,j,:,0].shape: ',boxes[i,j,:,0].shape)
                boxes[i,j,:,0] += node_x
                boxes[i,j,:,1] += node_y
        #print('boxes.shape: ',boxes.shape)#([8, 25, 8, 5])
        boxes = boxes.flatten(1,2)
        #print('boxes.shape:',boxes.shape)
        boxes = boxes.detach().cpu().numpy()
        return boxes

def getboxes_fromsample(pred,mask):
        box = pred[:,:,:,:5]
        label = pred[:,:,:,5]
        label = torch.where(label <= 0.5, 0.0, 1.0)
        label = F.sigmoid(label)

        label_index = label.unsqueeze(-1).repeat(1,1,1,5)
        #print('label_index.shape: ',label_index.shape)
        boxes = box*label_index
        mask = mask.float()
        #print('mask: ',mask)
        boxes = boxes*mask[:,:,:,:5]
        center=[]
        center_x = 0
        center_y = 0
        length_patch = 1/args.patch
        for i in range(args.patch**2):
            center.append(torch.tensor([center_x, center_y]))
            if (i+1) %args.patch == 0:
                center_y+=length_patch
                center_x = 0
            else:
                center_x+=length_patch

        for i in range(boxes.shape[0]):
            for j in range(boxes.shape[1]):
                node_x,node_y = center[j]
                #print('boxes[i,j,:,0].shape: ',boxes[i,j,:,0].shape)
                boxes[i,j,:,0] += node_x
                boxes[i,j,:,1] += node_y
        #print('boxes.shape: ',boxes.shape)#([8, 25, 8, 5])
        boxes = boxes.flatten(1,2)
        #print('boxes.shape:',boxes.shape)
        boxes = boxes.detach().cpu().numpy()
        return boxes


def main(args):

    #print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    #print("{}".format(args).replace(', ', ',\n'))
    color_white = (255,255,255)
    color_red = (0,0,255)
    device = torch.device(args.device)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_valid = ValidDataset(args.data_path,train=False,split_ratio = 0.8)

    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        #log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_valid = torch.utils.data.DataLoader(
        dataset_valid, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    # define the model
    model = MaskedAutoencoderViT(mask_ratio = args.mask_ratio, 
                                device = args.device, activation=args.activation, drop_ratio=args.drop_ratio,
                                depth=args.trans_deep*3, num_heads=args.num_heads, embed_dim=args.embed_dim, decoder_embed_dim = args.decoder_embed_dim,
                                decoder_depth=args.trans_deep, decoder_num_heads=args.num_heads, patch_size = args.patch)
    pretrained_model = torch.load(args.model_path)
    model.load_state_dict(pretrained_model)
    model.to(device)

    image_num = 0
    for epoch in range(args.start_epoch, args.epochs):
        #model.train(True)
        model.eval()
    
        for data_iter_step, samples in enumerate(data_loader_valid):

            samples = samples.to(device)

            loss, pred, mask, loss_giou, loss_class, loss_l1 = model(samples, mask_ratio=args.mask_ratio)
             
            boxes = getboxes_frompred(pred, mask)
            sample_boxes = getboxes_fromsample(samples, mask)
            masked_sample = getboxes_fromsample(samples, ~mask)

            for i in range(boxes.shape[0]):
                boxes_one = boxes[i]
                sample_boxes_one = sample_boxes[i]
                masked_sample_one = masked_sample[i]
                img_pre = state2img(box = boxes_one,color = color_red)
                img_sample = state2img(box = sample_boxes_one,color = color_red)
                img_sample_nomasked = state2img(box = masked_sample_one,color = color_white)
                sample_img_cat = cv2.addWeighted(img_sample_nomasked,1,img_sample,1,0)
                pred_img_cat = cv2.addWeighted(img_sample_nomasked,1,img_pre,1,0)


                img=cv2.hconcat([sample_img_cat,pred_img_cat])
                dir_path = args.save_path
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                image_num += 1
                if args.loss_name == 'giou':
                    cv2.imwrite(f'{dir_path}'+str(image_num)+'loss_giou:'+str(loss_giou.item())+'loss_class:'+str(loss_class.item())+'loss_l1:'+str(loss_l1.item())+'.jpg',img)
                elif args.loss_name == 'chamfer':
                    cv2.imwrite(f'{dir_path}'+str(image_num)+'loss_chamfer:'+str(loss.item())+'.jpg',img)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
