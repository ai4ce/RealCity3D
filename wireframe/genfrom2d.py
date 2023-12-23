from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
import torch

from dataset import *
from model_vertex import *
import argparse
from train_vertex import *

from model_face import ModelFace

def quantize_polys(polylist):
    # polylist: a list of tensor
    outpoly = []
    for poly in polylist:  
        x_min = torch.min(poly[:,0])  
        x_max = torch.max(poly[:,0])  
        y_min = torch.min(poly[:,0])  
        y_max = torch.max(poly[:,0])  
        scale = torch.max((x_max-x_min, y_max-y_min))
        center = torch.cat([(x_min/2+x_max/2).unsqueeze(0).unsqueeze(0), (y_min/2+y_max/2).unsqueeze(0).unsqueeze(0)], dim = -1)
        
        polynew = (poly-center)/scale + 0.5 # [0,1]
        polynew = (polynew*256).floor() #{0,1,2,...,255}

        poly_discardrepet = []
        for j in range(polynew.shape[0]):
            if polynew[j]==polynew[j-1]:
                continue
            poly_discardrepet.append(polynew[j])

        

    return outpoly

if __name__ == "__main__":
    basedir = os.path.dirname(os.path.abspath(__file__))
    print(basedir)

    device = torch.device(args.device)

    model_v = VertexModel(embedding_dim=256, num_layers=4, hidden_size=256)
    model_v.load_state_dict(torch.load("../results/model/model_wireframe/mae_reconstruction_best_vertex"))
    
    model_f = ModelFace(latent_dim = 256, num_layer = 4, device = device)
    model_f.load_state_dict(torch.load("../results/model/model_wireframe/mae_reconstruction_best_face"))  

    model_v.to(device)
    model_f.to(device)
    model_v.eval()
    model_f.eval()
    
    # 0.obj, 底面（第一个面）
    # vertices = []   # each: [z,y,x] float
    # faces = []      # each: [v1, v2, ...]   int
    # with open("Datasets/Brooklyn/10007.obj", "r") as lines:
    #     while 1:
    #         line = lines.readline()
    #         if line == "":
    #             break
    #         line = line.split(" ")
    #         if line[0] == "v":
    #             vertices.append(list(map(int, [line[3], line[2], line[1]])))
    #         if line[0] == "f":
    #             faces = list(map(int, line[1:]))
    #             break
    for i in range(30):
        prefix = []
        #print("V, F:")
        #print(vertices)
        #print(faces)
        # print(faces)
        # prefix_list = vertices[:max(faces)]
        # for t in prefix_list:
        #     prefix += t
        # print(prefix)
        model_v.generate(prefix, "models/onlyV.obj")

        
        
        vertices22 = []   # each:[x,y,z]int, as face model's input
        with open("models/onlyV.obj", "r") as op:
            while 1:
                line = op.readline().split()
                if not line:
                    break
                vertices22.append(list(map(int, [line[1],line[2],line[3]])))
        print(torch.tensor([vertices22], device='cuda').shape)  # [1, 39, 3]
        #print(torch.tensor([vertices22], device='cuda'))
        model_f.generate(f"models/RecoveredMesh_{i}.obj", torch.tensor([vertices22], device='cuda').float())   # must .float()
