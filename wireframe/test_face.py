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

from model_face import ModelFace
from dataset import Dataset_F

import cv2


def get_args_parser():
    parser = argparse.ArgumentParser('PolyGen pre-training', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=1, type=int)
    # Dataset parameters
    parser.add_argument('--data_path', default='out/test_data', type=str,
                        help='dataset path')
    
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--num_workers', default=20, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')

    parser.add_argument('--split_ratio', type = float, default=0.0)

    parser.add_argument('--num_layer', type = int, default=4)

    parser.add_argument('--embed_dim', type = int, default=256)
    parser.add_argument('--latent_dim', type = int, default=256)

    parser.set_defaults(pin_mem=True)

    return parser


def main(args):
    device = torch.device(args.device)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_valid = Dataset_F(args.data_path, train=False, split = args.split_ratio, device = device)

    data_loader_valid = torch.utils.data.DataLoader(
        dataset_valid, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=dataset_valid.collate_fn,
    )
    
    # define the model
    model_path = "../results/model/model_wireframe/mae_reconstruction_best_face"
    model = ModelFace(latent_dim = args.latent_dim, num_layer = args.num_layer, device = device)
    pretrained_model = torch.load(model_path)
    model.load_state_dict(pretrained_model)
    
    model.to(device)

    for epoch in range(args.epochs):
        for valid_step, sample in enumerate(data_loader_valid):
            # if valid_step > 10:
            #     break

            print(valid_step)

            model.eval()

            source_v = sample["source_v"].to(device)
            
            with torch.no_grad():
                out = model.generate(f"../results/test/facemodel/out_{valid_step}.obj", source_v, firstface=[])



if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    main(args)
