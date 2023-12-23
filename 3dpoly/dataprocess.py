import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
import shapely
from shapely.geometry import Polygon
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import leaves_list
from shapely.ops import unary_union
import cv2


datain = np.load('../datasets/3Dpoly/poly_np.npy') #[40000, 60, 20, 2]
datain_pos = np.load('../datasets/3Dpoly/polypos_np.npy') #[40000, 60, 2]
datain_info = np.load('../datasets/3Dpoly/polyinfo_np.npy') #[40000, 61]
datain_h = np.load('../datasets/3Dpoly/polyh_np.npy') #[40000,  60]

for batch in range(10):
    print(batch)
    vertices = []
    face = []
    data = datain[batch]
    info = datain_info[batch]
    h = datain_h[batch]
    num_vert = 0
    for i in range(int(info[0])):
        len = int(info[i+1])
        for k in range(len):
            vertices.append(list(data[i, k])+[0])
            vertices.append(list(data[i, k])+[h[i]])
            if k == 0:
                face.append([2*k+1+num_vert, 2*k+2+num_vert, 2*len+num_vert, 2*len-1+num_vert])
            else:
                face.append([2*k+1+num_vert, 2*k+2+num_vert, 2*k+num_vert, 2*k-1+num_vert])
        face.append([2*j+1+num_vert for j in range(len)])
        face.append([2*j+2+num_vert for j in range(len)])
        num_vert += 2*len

    file = open(f'../datasets/3Dwireframe/{batch}.obj', "w")
    for v in vertices:
        file.write("v "+f"{v[0]}"+" "+f"{v[1]}"+" "+f"{v[2]}")
        file.write("\n")
    for f in face:
        face = "f "
        for id in f:
            face += f"{id}"+" "
        file.write(face)
        file.write("\n")


        
