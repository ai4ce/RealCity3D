from torch.utils.data import Dataset
import copy
import json
import os
import sys
import torch
from torch import nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(__file__))

cur_dir = os.path.dirname(os.path.abspath(__file__))


class Dataset_F(Dataset):
    def __init__(self,
                 data_path=os.path.join(os.path.dirname(cur_dir),
                                        "Datasets/Brooklyn_processed"),
                 split=0.8,
                 train = True,
                 device = "cuda"):
        
        self.data_path = data_path

        self.total_filename = os.listdir(data_path)
        if train:
            self.filename = self.total_filename[:int(split*len(self.total_filename))]
            print("train_set:", len(self.filename))
        else:
            self.filename = self.total_filename[int(split*len(self.total_filename)):len(self.total_filename)]
            print("valid_set:", len(self.filename))

        self.pad_v = 256
        self.pad_f = 1000

        self.device = device

    def __len__(self):
        return len(self.filename)

    # @profile
    def __getitem__(self, index):
        vertices = []
        faces = []
        with open(os.path.join(self.data_path, self.filename[index])) as lines:
            while True:
                line = lines.readline().split()
                if not line:
                    break
                if line[0] == "v":
                    vertices.append([int(float(line[1])), int(float(line[2])), int(float(line[3]))])
                if line[0] == "f":
                    for i in line[1:]:
                        faces.append(int(i)+1)
                    faces.append(0)
            faces.append(1)
        return {"vertices": torch.clamp(torch.tensor(vertices),0,255), 
                "length_v": len(vertices),
                "faces": torch.tensor(faces),
                "length_f": len(faces)}

    # @profile
    def collate_fn(self, samples):
  
        lens_v = [sample["length_v"] for sample in samples]
        lens_f = [sample["length_f"] for sample in samples]
        max_len_v = max(lens_v)
        max_len_f = max(lens_f)
        bsz = len(lens_v)
        source_v = torch.LongTensor(bsz, max_len_v, 3)
        source_v.fill_(self.pad_v)
        source_f = torch.LongTensor(bsz, max_len_f)
        source_f.fill_(self.pad_f)
        target = torch.ones_like(source_f) * self.pad_f

        for idx, sample in enumerate(samples):
            source_v[idx, :sample["length_v"], :] = sample['vertices']
            source_f[idx, :sample["length_f"]-1] = sample['faces'][:-1]
            target[idx, :sample["length_f"]] = sample['faces']

        return {
            "source_v": source_v.float(),
            "length_v": torch.tensor(lens_v),
            "source_f": source_f,
            "length_f": torch.tensor(lens_f),
            "target": target
        }


class Dataset_V(Dataset):
    def __init__(self,
                 data_path=os.path.join(os.path.dirname(cur_dir),
                                        "Datasets/Brooklyn_processed"),
                 split=0.8,
                 train = True,
                 device = "cuda"):
        
        self.data_path = data_path

        self.total_filename = os.listdir(data_path)
        if train:
            self.filename = self.total_filename[:int(split*len(self.total_filename))]
            print("train_set:", len(self.filename))
        else:
            self.filename = self.total_filename[int(split*len(self.total_filename)):len(self.total_filename)]
            print("valid_set:", len(self.filename))

        self.pad_v = 258
        self.device = device

    def __len__(self):
        return len(self.filename)

    # @profile
    def __getitem__(self, index):
        vertices = [257, ]
        with open(os.path.join(self.data_path, self.filename[index])) as lines:
            while True:
                line = lines.readline().split()
                if not line:
                    break
                if line[0] == "v":
                    vertices.append(float(line[3]))
                    vertices.append(float(line[2]))
                    vertices.append(float(line[1]))
            vertices.append(256)     # end 256
        return {"vertices": torch.clamp(torch.tensor(vertices), 0, 300), 
                "length_v": len(vertices)}

    # @profile
    def collate_fn(self, samples):
  
        lens_v = [sample["length_v"] for sample in samples]
        max_len_v = max(lens_v)
        bsz = len(lens_v)
        source_v = torch.LongTensor(bsz, max_len_v)
        source_v.fill_(self.pad_v)
        target = self.pad_v * torch.ones_like(source_v)

        for idx, sample in enumerate(samples):
            source_v[idx, :sample["length_v"]] = sample['vertices'][:]
            target[idx, :sample["length_v"]-1] = sample['vertices'][1:]

        return {
            "source": source_v,
            "target": target
        }