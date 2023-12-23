from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
import torch

from dataset import *
from model_vertex import *
import argparse
from train_vertex import *

from model_face import ModelFace

if __name__ == "__main__":
    basedir = os.path.dirname(os.path.abspath(__file__))
    print(basedir)
    args = get_args_parser()
    args = args.parse_args()
    args.embedding_dim = 256
    args.hidden_size = 256
    args.device = "cuda"

    model = VertexModel()

    model.load_state_dict(torch.load(f"../results/model/model_wireframe/mae_reconstruction_best_vertex"))
    model = model.cuda()
    model.eval()

    device = torch.device(args.device)
    model_f = ModelFace(latent_dim = args.hidden_size, num_layer = args.num_layers, device = device)
    pretrained_model = torch.load("../results/model/model_wireframe/mae_reconstruction_best_face")
    model_f.load_state_dict(pretrained_model)
    model_f.to(device)
    model_f.eval()
    
    
    for i in range(30):     # each: [v1, v2, ...]   int
        vertices = []   # each: [z,y,x] float
        faces = [] 
        with open(f"../results/test/result_condition_wireframe/refer/{i}.obj", "r") as lines:
            while 1:
                line = lines.readline()
                if line == "":
                    break
                line = line.split(" ")
                if line[0] == "v":
                    vertices.append(list(map(int, [line[3], line[2], line[1]])))
                if line[0] == "f":
                    faces = list(map(int, line[1:]))
                    break
    
        prefix = []
        print(i)
        print(max(faces))
        prefix_list = vertices[:max(faces)]
        for t in prefix_list:
            prefix += t
        print(prefix)
        model.generate(prefix, f"../results/test/result_condition_wireframe/new/onlyV_{i}.obj")

        
        
        vertices22 = []   # each:[x,y,z]int, as face model's input
        with open(f"result/new/onlyV_{i}.obj", "r") as op:
            while 1:
                line = op.readline().split()
                if not line:
                    break
                vertices22.append(list(map(int, [line[1],line[2],line[3]])))
        print(torch.tensor([vertices22], device='cuda').shape)  # [1, 39, 3]
        #print(torch.tensor([vertices22], device='cuda'))
        model_f.generate(f"../results/test/result_condition_wireframe/new/RecoveredMesh_{i}.obj", torch.tensor([vertices22], device='cuda').float(), faces)   # must .float()