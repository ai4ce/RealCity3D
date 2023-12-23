import argparse

import numpy as np

import os
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

from models_mage import MAGECityGen

from dataloader import ValidDataset
import math
import numpy as np
import random

import torch
import os
import cv2

import torch.nn.functional as F


def draw_polygon_c(img, point, txt, center, color):
    
    cv2.line(img, tuple(point[1]), tuple(point[0]), color, 1)
    cv2.line(img, tuple(point[1]), tuple(point[2]), color, 1)
    cv2.line(img, tuple(point[2]), tuple(point[3]), color, 1)
    cv2.line(img, tuple(point[0]), tuple(point[3]), color, 1)
    return img

def rotate_xy(p, sin, cos, center):
    x_ = (p[:,0:1]-center[:,0:1])*cos-(p[:,1:2]-center[:,1:2])*sin+center[:,0:1]
    y_ = (p[:,0:1]-center[:,0:1])*sin+(p[:,1:2]-center[:,1:2])*cos+center[:,1:2]

    return np.hstack((x_, y_))

def state2img(box,color):

    X_tree0 = box[:,:2]
    Feature_tree0 = box[:, 2:]
    alf = Feature_tree0[:,2:]

    cosO = np.cos(alf)
    sinO = np.sin(alf)

    ld = np.hstack((X_tree0[:,0:1]-Feature_tree0[:,0:1]/2, X_tree0[:,1:2]-Feature_tree0[:,1:2]/2))
    rd = np.hstack((X_tree0[:,0:1]+Feature_tree0[:,0:1]/2, X_tree0[:,1:2]-Feature_tree0[:,1:2]/2))
    ru = np.hstack((X_tree0[:,0:1]+Feature_tree0[:,0:1]/2, X_tree0[:,1:2]+Feature_tree0[:,1:2]/2))
    lu = np.hstack((X_tree0[:,0:1]-Feature_tree0[:,0:1]/2, X_tree0[:,1:2]+Feature_tree0[:,1:2]/2))
    ld_r = rotate_xy(ld, sinO, cosO, X_tree0)
    rd_r = rotate_xy(rd, sinO, cosO, X_tree0)
    ru_r = rotate_xy(ru, sinO, cosO, X_tree0)
    lu_r = rotate_xy(lu, sinO, cosO, X_tree0)

    box_r = np.hstack((ld_r*300, rd_r*300, ru_r*300, lu_r*300)).reshape(len(X_tree0), -1, 2)

    img = np.ones(shape=(400,400,3), dtype=np.int16)

    for j, p in enumerate(box_r):
        p = np.array(p).astype(int)
        img = draw_polygon_c(img,p, j, X_tree0[j,:2],color)
    return img


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=1, type=int)

    parser.add_argument('--mask_ratio', default=0.5, type=float,
                        help='Masking ratio (percentage of removed patches).')

    # Dataset parameters
    parser.add_argument('--data_path', default='../results/states/states_d50.npy', type=str,
                        help='dataset path')
    
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')

    parser.add_argument('--activation', default='sigmoid')
    parser.add_argument('--split_ratio', type = float, default=0.8)
    parser.add_argument('--trans_deep', type = int, default=8)
    parser.add_argument('--trans_deep_decoder', type = int, default=4)
    parser.add_argument('--num_heads', type = int, default=8)

    parser.add_argument('--embed_dim', type = int, default=256)
    parser.add_argument('--decoder_embed_dim', type = int, default=64)

    parser.add_argument('--drop_ratio', type = float, default=0.0)

    parser.add_argument('--model_path', type = str, default='/home/rl4citygen/DRL4CityGen/results/magetest/output_dir/maged_pos20/mae_reconstruction_best.pth')
  
    parser.add_argument('--save_path', type = str, default="../results/statedraw_50/")
    parser.add_argument('--use_trainset', type = bool, default=False)
    parser.add_argument('--translation', type = float, default=0.5)
    parser.add_argument('--pos_weight', type = int, default=10)
    parser.set_defaults(pin_mem=True)

    return parser




def main(args):

    color_white = (255,255,255)
    color_red = (0,0,255)
    device = torch.device(args.device)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_test = ValidDataset(args.data_path,train=args.use_trainset,split_ratio = 0.8)


    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    # define the model
    model = MAGECityGen(drop_ratio = args.drop_ratio, pos_weight=args.pos_weight,
                        device = args.device, activation=args.activation, num_heads=args.num_heads,
                        depth=args.trans_deep, embed_dim=args.embed_dim, decoder_embed_dim = args.decoder_embed_dim,
                        decoder_depth=args.trans_deep_decoder, decoder_num_heads=args.num_heads)
    
    pretrained_model = torch.load(args.model_path)
    model.load_state_dict(pretrained_model)
    model.to(device)

    image_num = 0
    model.eval()
    
    for data_iter_step, samples in enumerate(data_loader_test):

        for i in range(1):
            samples = samples.to(device)

            model.eval()

            assert samples.shape[0] == 1

            _, pred, mask, _, _, _ = model(samples, mask_ratio=args.mask_ratio)

            sample_ = samples[0][:,:5]
            mask_ = mask[0]
            pred_ = pred[0]
            reserved_samples = (~mask_* sample_).detach().cpu().numpy()  
            masked_samples = (mask_* sample_).detach().cpu().numpy()

            ###############
            mask_sort = ~mask_[:,0].unsqueeze(-1).repeat(1,2)
            reserve_sort = torch.masked_select(samples[0][:,5:], mask_sort).view(-1, 2) 
            unmask_samples = torch.masked_select(samples[0], ~mask_[:,0].unsqueeze(-1).repeat(1,7)).view(-1, 7)
            remain_x = []
            remain_y = []
            for i in range(32):
                if i not in reserve_sort[:, 0]:
                    remain_x.append(i)
                if i not in reserve_sort[:, 1]:
                    remain_y.append(i)
            prob_pred = torch.sigmoid(pred_[:, 5])

            prob_remain = torch.zeros(prob_pred.shape)
            for r_x in remain_x:
                for r_y in remain_y:
                    prob_remain[r_x*32+r_y] = prob_pred[r_x*32+r_y]
                            
            id_sam = torch.multinomial(prob_remain, 1).item()
            x_max = id_sam//32
            y_max = id_sam%32

            max_prob = 0
            # for r_x in remain_x:
            #     for r_y in remain_y:
            #         if prob_pred[r_x*32+r_y] > max_prob:
            #             max_prob = prob_pred[r_x*32+r_y]
            #             x_max = r_x
            #             y_max = r_y
                        
            remain_x.remove(x_max)
            remain_y.remove(y_max)
            print(len(remain_x), len(remain_y), max_prob)
            max_pred = torch.cat([pred_[x_max*32+y_max, :5],torch.tensor([x_max, y_max]).to(device)], dim = 0).unsqueeze(0)
            
            samples = torch.cat([unmask_samples, max_pred], dim = 0).unsqueeze(0).detach()
            print(samples.shape)

            for i in range(int(32*args.mask_ratio)-1):
                _, pred, _, _, _, _ = model(samples, mask_ratio=0, iter = True)

                sample_ = samples[0][:,:5]
 
                pred_ = pred[0]

                prob_pred = torch.sigmoid(pred_[:, 5])

                prob_remain = torch.zeros(prob_pred.shape)
                for r_x in remain_x:
                    for r_y in remain_y:
                            prob_remain[r_x*32+r_y] = prob_pred[r_x*32+r_y]

                id_sam = torch.multinomial(prob_remain, 1).item()
                x_max = id_sam//32
                y_max = id_sam%32

                # max_prob = 0
                # for r_x in remain_x:
                #     for r_y in remain_y:
                #         if prob_pred[r_x*32+r_y] > max_prob:
                #             max_prob = prob_pred[r_x*32+r_y]
                #             x_max = r_x
                #             y_max = r_y
                            
                remain_x.remove(x_max)
                remain_y.remove(y_max)
                print(len(remain_x), len(remain_y), max_prob)
                max_pred = torch.cat([pred_[x_max*32+y_max, :5],torch.tensor([x_max, y_max]).to(device)], dim = 0).unsqueeze(0)

                samples = torch.cat([samples[0], max_pred], dim = 0).unsqueeze(0).detach()
                print(samples.shape)



            masked_pred = samples[0, int((1-args.mask_ratio)*32):, :5].detach().cpu().numpy()  
                
            img_reserved_samples = state2img(box=reserved_samples,color = color_white)
            img_masked_samples= state2img(box = masked_samples,color = color_red)
            img_pred = state2img(box = masked_pred,color = color_red)
            img_box_pred = cv2.addWeighted(img_reserved_samples,1,img_pred,1,0)
            img_box_true = cv2.addWeighted(img_reserved_samples,1,img_masked_samples,1,0)
                
            img=cv2.hconcat([img_box_true,img_box_pred])
            dir_path = args.save_path
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            image_num += 1
                
            cv2.imwrite(f'{dir_path}'+str(image_num) +'.jpg',img)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)