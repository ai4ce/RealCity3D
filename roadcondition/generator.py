import argparse

import numpy as np

import os
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

from models_mage import MAGECityPolyGen
from pospred.pospred_model import MAGECityPosition_Minlen

from dataloader import PolyDataset
import math
import numpy as np
import random
import json

import torch
import os
import cv2
import torch.nn.functional as F

from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union
import numpy as np

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

        return poly_reserve, pos_reserve, pos_tar

def in_poly_idx(poly, discre = 100):
    """
    :param p: [x, y]
    :param poly: [[], [], [], [], ...]
    :return:
    """

    points = torch.cat([torch.arange(discre).repeat_interleave(discre).unsqueeze(-1), torch.arange(discre).repeat(discre).unsqueeze(-1)], dim = -1)
    index = -torch.ones(discre*discre)
    for i, corner in enumerate(poly):
        next_i = i + 1 if i + 1 < len(poly) else 0
        x1, y1 = corner
        x2, y2 = poly[next_i]
        condition1 =  (min(y1, y2) < points[:, 1])* (points[:, 1]<= max(y1, y2))  # find horizontal edges of polygon
        x = x1 + (points[:, 1] - y1) * (x2 - x1) / (y2 - y1)
        condition2 = x > points[:, 0]
        index = index*((-(condition1*condition2).long())*2+1)

    out = torch.where(index>0, 0, 1)
    return out 

def infgen(model, modelpos, road, samples_iter, pos_iter, road_poly, device, discard_prob = 0.1, size = 1000, discre = 100, use_sample=False, finetune = False):

    remain_flag = torch.ones(discre**2).to(device)
        
    endflag = 0
    gen_iter = []

    num_poly = 0

    for npoly in range(250):
        pred = modelpos(samples_iter, pos_iter, None, road, generate = True)
        prob_pred = torch.sigmoid(pred[0, :])
        prob_pred = torch.where(prob_pred < discard_prob, torch.tensor(0.0).to(device), prob_pred) 
        while True:
            prob_pred = prob_pred*remain_flag
            if torch.max(prob_pred) < discard_prob:
                endflag = 1
                break      
            if use_sample == False:
                idx_iter = torch.argmax(prob_pred)
            else:            
                idx_iter = torch.multinomial(prob_pred, 1).squeeze(0)
                                
            remain_flag[idx_iter] = 0

            pred_pos = torch.cat([idx_iter.unsqueeze(0)//discre, idx_iter.unsqueeze(0)%discre],dim=0).unsqueeze(0).unsqueeze(0)
            predpoly = model.infgen(samples_iter, pos_iter, pred_pos, road)[0]

            polygon = Polygon(np.array(predpoly[0].clone().detach().cpu()))
            intersect_flag = 0
            for pe in range(samples_iter.shape[1]):
                polyexist = [] 
                for k in range(samples_iter.shape[2]):
                    if samples_iter[0, pe, k, 0] != 0:
                        point = samples_iter[0, pe, k, :].clone().detach().cpu().numpy()
                        polyexist.append(point)
                # print(polygon, polyexist)
                polyexist = Polygon(polyexist)
                polygon = polygon.buffer(0)
                polyexist = polyexist.buffer(0)
                if polygon.intersection(polyexist).area>10:
                    intersect_flag = 1
                    break

            for polyexist in road_poly:
                polygon = polygon.buffer(0)
                polyexist = polyexist.buffer(0)
                if polygon.intersection(polyexist).area>10:
                    intersect_flag = 1
                    break

            if intersect_flag == 0:
                break

        if endflag == 1:
            break
        
        poly_add, pos_add = pad_poly(predpoly)
        poly_add = poly_add.to(device)
        pos_add = pos_add.to(device)
        samples_iter = torch.cat([samples_iter, poly_add], dim = 1).detach()
        pos_iter = torch.cat([pos_iter, pos_add], dim = 1).detach()
        num_poly+=1

        gen_iter.append(predpoly.squeeze(0).detach())
        remain_flag = remain_flag*in_poly_idx(predpoly.squeeze(0).detach().cpu()).to(device)

    if finetune:
        gen_iter = model.genfirst(pos_iter)
        gen_iter = [genpoly.squeeze(0).detach() for genpoly in gen_iter]
 
    print("gen:", len(gen_iter))
    return gen_iter

def pad_poly(poly_list, max_l = 20):
    num = len(poly_list)
    
    pad_poly = torch.zeros([num, 20, 2])
    pos_in = torch.zeros([num, 2])
    for i in range(num):
        pad_poly[i, :poly_list[i].shape[0], :] = poly_list[i]
        pos_in[i,:] = torch.mean(poly_list[i], dim = 0)//10
    
    return pad_poly.unsqueeze(0), pos_in.unsqueeze(0)


def road_to_polygon(road, offset_distance = 20):
    poly = []
    for line in road[0, :int(np.where(road[0, :, 0, 0]+road[0, :, 0, 1] == 0)[0][0])]:
        if np.where(line[:, 0]+line[:, 1] == 0)[0].size > 0:
            line = LineString(np.array(line[:int(np.where(line[:, 0]+line[:, 1] == 0 )[0][0])], np.int32))
        else:
            line = LineString(np.array(line, np.int32))
        offset_line = line.parallel_offset(offset_distance, 'right')
        offset_line_ = line.parallel_offset(offset_distance, 'left')
        poly.append(Polygon(np.concatenate([np.array(offset_line_.coords), np.flip(np.array(offset_line.coords), axis = 0)], axis = 0)))
    return poly

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=1, type=int)

    # Dataset parameters
    parser.add_argument('--data_path', default='../datasets/dataroad/poly_np.npy', type=str,
                        help='dataset path')
    parser.add_argument('--datapos_path', default='../datasets/dataroad/polypos_np.npy', type=str,
                        help='dataset path')
    parser.add_argument('--datainfo_path', default='../datasets/dataroad/polyinfo_np.npy', type=str,
                        help='dataset path')
    parser.add_argument('--dataroad_path', default='../datasets/dataroad/polyroad_np.npy', type=str,
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
    parser.add_argument('--trans_deep', type = int, default=12)
    parser.add_argument('--trans_deep_decoder', type = int, default=8)
    parser.add_argument('--num_heads', type = int, default=8)


    parser.add_argument('--embed_dim', type = int, default=512)
    parser.add_argument('--decoder_embed_dim', type = int, default=512)

    parser.add_argument('--drop_ratio', type = float, default=0.0)
    parser.add_argument('--pos_weight', type = int, default=20)
    parser.add_argument('--max_build', type = int, default=250)
    parser.set_defaults(pin_mem=True)

    # parser.add_argument('--model_path', type = str, default='/home/rl4citygen/InfiniteCityGen/results/model/model_auto/64_800_12_8_8_512_256_0.1_mae_reconstruction.pth')
    parser.add_argument('--model_path', type = str, default='/scratch/rx2281/pytorch/InfiniteCityGen/results/train/polyautoroad_append/output_dir/64_1000_12_8_8_512_512_0.1_1e-3/mae_reconstruction_best.pth')
    parser.add_argument('--save_path', type = str, default="../results/test/states_roadcond_append/")
    parser.add_argument('--use_trainset', type = bool, default=False)
    parser.add_argument('--translation', type = float, default=0.5)

    return parser


def main(args):

    color_white = (255,255,255)
    color_red = (0,0,255)
    device = torch.device(args.device)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_test = PolyDataset(args.data_path, args.datapos_path, args.datainfo_path, args.dataroad_path, train=False,split_ratio = args.split_ratio)


    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    

    model = MAGECityPolyGen(drop_ratio = args.drop_ratio, num_heads=args.num_heads, device = args.device, max_build = args.max_build,
                        depth=args.trans_deep, embed_dim=args.embed_dim, decoder_embed_dim = args.decoder_embed_dim,
                        decoder_depth=args.trans_deep_decoder, decoder_num_heads=args.num_heads)
    
    pretrained_model = torch.load(args.model_path)
    model.load_state_dict(pretrained_model)
    model.to(device)
    model.eval()

    modelpos = MAGECityPosition_Minlen(drop_ratio = 0.0, num_heads=args.num_heads, device = args.device,
                            depth=6, embed_dim=512, decoder_embed_dim = 16,
                            discre=100, patch_num=10, patch_size=10,
                            decoder_depth=3, decoder_num_heads=args.num_heads, pos_weight = 100)
    
    pretrained_modelpos = torch.load("/scratch/rx2281/pytorch/InfiniteCityGen/results/train/polypospredroad_append/output_dir/128_3000_6_3_8_512_16_0.1_pos100_1e-3/mae_reconstruction_best.pth")
    modelpos.load_state_dict(pretrained_modelpos)
    modelpos.to(device)
    modelpos.eval()

    image_num = 0
    
    assert args.batch_size == 1
    use_s = False

    list_count=[]
    list_ratio=[]
    list_area=[]
    list_edge=[]

    list_poly=[]
    unvalid = 0
    total_count = 0

    size = 1000

    for valid_step, (samples, pos, info, road) in enumerate(data_loader_test):
        if valid_step >=500:
            break
        
        road_poly = road_to_polygon(road)
        road = road.to(device).float()
        poly_reserve, pos_reserve, pos_tar = random_masking(samples, pos, info, 0)

        poly_reserve = poly_reserve.to(device)
        pos_reserve = pos_reserve.to(device)
        
        gen_poly = infgen(model = model, modelpos = modelpos, road = road, samples_iter = poly_reserve, pos_iter = pos_reserve, road_poly = road_poly, device = device,use_sample=use_s)
    
        
        total_area=0
        list_count.append(len(gen_poly))
        list_poly.append(np.array([np.array(poly.cpu()) for poly in gen_poly], dtype=object))

        img = np.ones((size,size,3),np.uint8)*255

        road = road.detach().cpu()
        r = road[0, :int(np.where(road[0, :, 0, 0]+road[0, :, 0, 1] == 0)[0][0])]
        for ro in r:
            if np.where(ro[:, 0]+ro[:, 1] == 0)[0].size > 0:
                string = np.array(ro[:int(np.where(ro[:, 0]+ro[:, 1] == 0 )[0][0])], np.int32)
            else:
                string = np.array(ro, np.int32)
            cv2.polylines(img,[string],False,(128, 0, 0),thickness=20)
        
        for num, poly in enumerate(gen_poly):
            total_count+=1
            
            pts = np.array(poly.cpu(), np.int32)
            pts = pts.reshape((-1,1,2)).astype(int)
            list_edge.append(len(poly))
            if Polygon(poly.cpu()).is_valid:
                areatem = Polygon(poly.cpu()).area
                list_area.append(areatem)
                total_area+= areatem
            else:
                unvalid+=1

            cv2.fillPoly(img, [pts], color=(255, 255, 0))
        list_ratio.append(total_area/size**2)

        for num, poly in enumerate(gen_poly):
            pts = np.array(poly.cpu(), np.int32)
            pts = pts.reshape((-1,1,2)).astype(int)

            cv2.polylines(img,[pts],True,(0,0,0),1)
    
        dir_path = args.save_path+'img/'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        image_num += 1
        print(image_num)

        # print(torch.where(torch.tensor(img[:, :, 2])==255, 0, 1).sum()/total_area)
                
        cv2.imwrite(f'{dir_path}'+str(image_num) +'.jpg',img)
    
    dict_100 = {'count': list_count,
                'area': list_area,
                'edge': list_edge,
                "ratio":list_ratio}

    with open(f'{args.save_path}'+'wd.json', 'w') as json_file:
        json.dump(dict_100, json_file)

    np.save(f'{args.save_path}'+'poly.npy', np.array(list_poly, dtype=object))
    print(unvalid/total_count)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)

