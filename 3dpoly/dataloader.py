import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import os
import numpy as np

import torch.nn.functional as F
class PolyDataset3D(Dataset):
    def __init__(self, data_path, datapos_path, datainfo_path, datah_path, train = True, split_ratio = 0.8):

        self.data_path = data_path
        self.datapos_path = datapos_path
        self.datainfo_path = datainfo_path
        self.datah_path = datah_path
        self.datas = np.load(self.data_path)
        self.datapos = np.load(self.datapos_path)
        self.datainfo = np.load(self.datainfo_path)
        self.datah = np.load(self.datah_path)
        self.split_ratio = split_ratio
        if train:
            self.datas = self.datas[0:int(len(self.datas)*self.split_ratio)]
            self.datapos = self.datapos[0:int(len(self.datapos)*self.split_ratio)]
            self.datainfo = self.datainfo[0:int(len(self.datainfo)*self.split_ratio)]
            self.datah = self.datah[0:int(len(self.datah)*self.split_ratio)]
            print('train_len(self.datas): ',len(self.datas))
            print('train_len(self.datapos): ',len(self.datapos))
            print('train_len(self.datainfo): ',len(self.datainfo))
            print('train_len(self.datah): ',len(self.datah))
        else:
            self.datas = self.datas[int(len(self.datas)*self.split_ratio):int(len(self.datas))]
            self.datapos = self.datapos[int(len(self.datapos)*self.split_ratio):int(len(self.datapos))]
            self.datainfo = self.datainfo[int(len(self.datainfo)*self.split_ratio):int(len(self.datainfo))]
            self.datah = self.datah[int(len(self.datah)*self.split_ratio):int(len(self.datah))]
            print('test_len(self.datas): ',len(self.datas))
            print('test_len(self.datapos): ',len(self.datapos))
            print('test_len(self.datainfo): ',len(self.datainfo))
            print('test_len(self.datah): ',len(self.datah))

    def __getitem__(self, index):
        data =self.datas[index]
        datapos = self.datapos[index]
        datainfo = self.datainfo[index]
        datah = self.datah[index]
        return data, datapos, datainfo, datah

    def __len__(self):
        return len(self.datas)






