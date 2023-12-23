import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import os
import numpy as np

import torch.nn.functional as F
class ValidDataset(Dataset):
    def __init__(self,root_path,train = True,split_ratio = 0.8):

        self.root_path = root_path
        self.datas = np.load(self.root_path)
        self.split_ratio = split_ratio
        if train:
            self.datas = self.datas[0:int(len(self.datas)*self.split_ratio)]
            print('train_len(self.datas): ',len(self.datas))
        else:
            self.datas = self.datas[int(len(self.datas)*self.split_ratio):int(len(self.datas))]
            print('test_len(self.datas): ',len(self.datas))

    def __getitem__(self, index):
        data =self.datas[index]
        data = torch.from_numpy(data)
        return data

    def __len__(self):
        return len(self.datas)

    # def getdata(self):
    #     data_path = []
    #     root = self.root_path
    #     data_name = os.listdir(root)
    #     for name in data_name:
    #         file_path = os.path.join(root,name)
    #         data_path.append(file_path)
    #     data_path = np.array(data_path)
    #     return data_path





