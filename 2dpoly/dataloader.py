import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import os
import numpy as np

import torch.nn.functional as F
class PolyDataset(Dataset):
    def __init__(self, data_path, datapos_path, datainfo_path, train = True,split_ratio = 0.8, data_aug = False):

        self.data_path = data_path
        self.data_aug = data_aug
        self.datapos_path = datapos_path
        self.datainfo_path = datainfo_path
        self.datas = np.load(self.data_path)
        self.datapos = np.load(self.datapos_path)
        self.datainfo = np.load(self.datainfo_path)
        self.split_ratio = split_ratio
        if train:
            self.datas = self.datas[0:int(len(self.datas)*self.split_ratio)]
            self.datapos = self.datapos[0:int(len(self.datapos)*self.split_ratio)]
            self.datainfo = self.datainfo[0:int(len(self.datainfo)*self.split_ratio)]
            print('train_len(self.datas): ',len(self.datas))
            print('train_len(self.datapos): ',len(self.datapos))
            print('train_len(self.datainfo): ',len(self.datainfo))
        else:
            self.datas = self.datas[int(len(self.datas)*self.split_ratio):int(len(self.datas))]
            self.datapos = self.datapos[int(len(self.datapos)*self.split_ratio):int(len(self.datapos))]
            self.datainfo = self.datainfo[int(len(self.datainfo)*self.split_ratio):int(len(self.datainfo))]
            print('test_len(self.datas): ',len(self.datas))
            print('test_len(self.datapos): ',len(self.datapos))
            print('test_len(self.datainfo): ',len(self.datainfo))


    def __getitem__(self, index):
        data =self.datas[index]
        datapos = self.datapos[index]
        datainfo = self.datainfo[index]
        if self.data_aug:
            rand_noise = torch.rand(2)
            if rand_noise[0]<0.5:
                data[:, :, 0] = 500-data[:, :, 0]
                datapos[:, 0] = 49-datapos[:, 0]
            if rand_noise[1]<0.5:
                data_tem = data[:, :, 0].copy()
                datapos_tem = datapos[:, 0].copy()
                data[:, :, 0] = data[:, :, 1].copy()
                datapos[:, 0] = datapos[:, 1].copy()
                data[:, :, 1] = data_tem
                datapos[:, 1] = datapos_tem
                
        return data, datapos, datainfo

    def __len__(self):
        return len(self.datas)






