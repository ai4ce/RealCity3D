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

import torch
import os
import cv2
from shapely.geometry import Polygon

import torch.nn.functional as F

def infgen_noiter(model, modelpos, samples, pos, device, x, y, use_sample=False):

    inlen = samples.shape[1]  
    print("in_len:", inlen)
    samples_iter = samples.clone()
    pos_iter = pos.clone()

    remain_flag = torch.ones([50,50]).to(device)
    if x!=0:
        remain_flag[:25, :] = 0
    if y!=0:
        remain_flag[:, :25] = 0
    remain_flag = remain_flag.flatten(0,1)
    for p in pos[0]:
        remain_flag[int(p[0]*50+p[1])] = 0
        

   
    pred = modelpos(samples_iter, pos_iter, None, generate = True)

    prob_pred = torch.sigmoid(pred[0, :])*remain_flag
    ptem, _ = torch.sort(prob_pred,descending=True)
    probthred = ptem[60-inlen]
    prob_pred = (prob_pred).view(50,50)
    index = torch.nonzero(torch.where(prob_pred>probthred, 1.0, 0.0))
    pos_tar = torch.cat([index[i].unsqueeze(0) for i in range(len(index))], dim = 0).unsqueeze(0)

    
    predpoly = model.infgen(samples_iter, pos_iter, pos_tar)

  
        # max_pred = pred[0, idx_iter, :]

        # leng = torch.argmax(max_pred[40:60].softmax(0))+1
        # print(leng)
        # pred_poly = max_pred[:40].view(20, 2)[:leng, :]
    gen_poly = []
    for poly in predpoly:
        gen_poly.append(poly.squeeze(0).detach())
    print("gen:", len(gen_poly))

    return gen_poly

def infgen(model, modelpos, samples, pos, device, x, y, discard_prob = 0.05, use_sample=False):

    inlen = samples.shape[1]  
    print("in_len:", inlen)
    samples_iter = samples.clone()
    pos_iter = pos.clone()

    remain_flag = torch.ones([50,50]).to(device)
    if x!=0:
        remain_flag[:25, :] = 0
    if y!=0:
        remain_flag[:, :25] = 0
    remain_flag = remain_flag.flatten(0,1)
    for p in pos[0]:
        remain_flag[int(p[0]*50+p[1])] = 0
        

    endflag = 0
    gen_iter = []

    num_poly = inlen

    for npoly in range(60 - inlen):
        pred = modelpos(samples_iter, pos_iter, None, generate = True)
        # pred_draw = torch.sigmoid(pred).view(50,50,1).permute(1,0,2).repeat(1,1,3).detach().cpu()
        # posimg_pred = np.array(pred_draw*255, dtype=np.uint8)

        # cv2.imwrite(f'./results/test/'+str(npoly)+ str(x)+ str(y)+'.jpg',posimg_pred)
        prob_pred = torch.sigmoid(pred[0, :])
        while True:
            prob_pred = prob_pred*remain_flag
            idx_iter = torch.argmax(prob_pred)
            max_prob = prob_pred[idx_iter]
            if max_prob < 0.1:
                endflag = 1
                break
            # if use_sample == False:
            #     idx_iter = torch.argmax(prob_pred)
            #     max_prob = prob_pred[idx_iter]
                
            #     if max_prob > first_prob:
            #         first_prob = max_prob
            #     if max_prob < first_prob - discard_prob:
            #         break
            # else:                 
            #     idx_iter = torch.multinomial(prob_pred, 1)
                                
            remain_flag[idx_iter] = 0

            pred_pos = torch.cat([idx_iter.unsqueeze(0)//50, idx_iter.unsqueeze(0)%50],dim=0).unsqueeze(0).unsqueeze(0)
            # print(pred_pos.shape)
            predpoly = model.infgen(samples_iter, pos_iter, pred_pos)[0]
            # print(predpoly.shape)

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
                if polygon.intersects(polyexist):
                    intersect_flag = 1
                    break

            if intersect_flag == 0:
                break
        if endflag == 1:
            break
        # max_pred = pred[0, idx_iter, :]

        # leng = torch.argmax(max_pred[40:60].softmax(0))+1
        # print(leng)
        # pred_poly = max_pred[:40].view(20, 2)[:leng, :]
        
        poly_add, pos_add = pad_poly(predpoly)
        poly_add = poly_add.to(device)
        pos_add = pos_add.to(device)
        samples_iter = torch.cat([samples_iter, poly_add], dim = 1).detach()
        pos_iter = torch.cat([pos_iter, pos_add], dim = 1).detach()
        num_poly+=1

        gen_iter.append(predpoly.squeeze(0).detach())
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

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--num_workers', default=20, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.add_argument('--activation', default='sigmoid')

    parser.add_argument('--split_ratio', type = float, default=0.8)
    parser.add_argument('--trans_deep', type = int, default=12)
    parser.add_argument('--trans_deep_decoder', type = int, default=8)
    parser.add_argument('--num_heads', type = int, default=8)


    parser.add_argument('--embed_dim', type = int, default=512)
    parser.add_argument('--decoder_embed_dim', type = int, default=256)

    parser.add_argument('--drop_ratio', type = float, default=0.0)
    parser.add_argument('--pos_weight', type = int, default=20)
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--model_path', type = str, default='/home/rl4citygen/DRL4CityGen/results/polyauto/output_dir/64_800_12_8_8_512_256_0.0_pos20_100/mae_reconstruction_best.pth')
    parser.add_argument('--save_path', type = str, default="../results/test/states_poly_draw_0.0/")
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

    dataset_test = PolyDataset(args.data_path, args.datapos_path, args.datainfo_path, train=False,split_ratio = args.split_ratio)


    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    


    model = MAGECityPolyGen(drop_ratio = args.drop_ratio, num_heads=args.num_heads, device = args.device,
                        depth=args.trans_deep, embed_dim=args.embed_dim, decoder_embed_dim = args.decoder_embed_dim,
                        decoder_depth=args.trans_deep_decoder, decoder_num_heads=args.num_heads)
    
    pretrained_model = torch.load(args.model_path)
    model.load_state_dict(pretrained_model)
    model.to(device)
    model.eval()

    modelpos = MAGECityPosition_Minlen(drop_ratio = 0.0, num_heads=args.num_heads, device = args.device,
                            depth=6, embed_dim=256, decoder_embed_dim = 32,
                            decoder_depth=3, decoder_num_heads=args.num_heads, pos_weight = 100)
    
    pretrained_modelpos = torch.load("../results/model/model_pospred/patchify_0.0_6_3_8_256_32.pth")
    modelpos.load_state_dict(pretrained_modelpos)
    modelpos.to(device)
    modelpos.eval()

    image_num = 0
    
    assert args.batch_size == 1
    inf_t = args.translation*500

    for data_iter_step, (ini_samples, _, ini_info) in enumerate(data_loader_test):
        if data_iter_step >= 50:
           break
        
        inf_num = 3
        use_s = False

        samples_draw = []

        for i in range(int(ini_info[0,0])):
            samples_draw.append(ini_samples[0, i, :int(ini_info[0,i+1])])
        
        
        for j in range(inf_num+1):
            for i in range(inf_num+1):   
                if i == 0 and j == 0:
                    continue  

                sample_in = []
                for poly in samples_draw:
                    if ((i*inf_t<poly[:, 0])*(poly[:, 0]<(i*inf_t + 500))*(j*inf_t<poly[:, 1])*(poly[:, 1]<(j*inf_t + 500))).all():
                        poly_ = poly.clone()
                        poly_[:, 0] -= i*inf_t
                        poly_[:, 1] -= j*inf_t
                        sample_in.append(poly_)

                sample_in, pos_in = pad_poly(sample_in)
                # print(sample_in, pos_in)
                sample_in = sample_in.to(device)
                pos_in = pos_in.to(device)
      
                gen_poly = infgen_noiter(model = model, modelpos = modelpos, samples = sample_in, pos = pos_in, device = device, x=i, y=j, use_sample=use_s, )

                for polybulid in gen_poly:
                    polydraw = polybulid.clone()
                    polydraw[:, 0] += i*inf_t
                    polydraw[:, 1] += j*inf_t
                    samples_draw.append(polydraw)
                print(len(samples_draw)) 
              
        
        img = np.zeros((int(500+inf_num*inf_t),int(500+inf_num*inf_t),3),np.uint8)

        for poly in samples_draw:
            pts = np.array(poly.cpu(), np.int32)
            pts = pts.reshape((-1,1,2)).astype(int)
            cv2.polylines(img,[pts],True,(0,255,255),1)
        
        dir_path = args.save_path
        if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        image_num += 1
        print(image_num)
                
        cv2.imwrite(f'{dir_path}'+str(image_num) +'.jpg',img)




if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)

