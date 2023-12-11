import numpy as np
import torch
from torchvision.datasets.mnist import MNIST
from torch.utils.data.dataset import Dataset
from PIL import Image, ImageDraw, ImageOps
import json
from torch.utils.data.dataloader import DataLoader
import cv2
from utils import trim_tokens, gen_colors
import os
import matplotlib.pyplot as plt

class Padding(object):
    def __init__(self, max_length,vocab_size):
        self.max_length = max_length
        self.bos_token = vocab_size - 3
        self.eos_token = vocab_size - 2
        self.pad_token = vocab_size - 1

    def __call__(self, layout):
        #print('layout: ',layout)
        # grab a chunk of (max_length + 1) from the layout
        #print('layout.shape: ',layout.shape)
        chunk = torch.zeros(self.max_length+1 , dtype=torch.long) #+ self.pad_token
        # Assume len(item) will always be <= self.max_length:
        chunk[0] = self.bos_token
        #print('chunk[0]: ',chunk[0])
        chunk[1:len(layout)+1] = layout
        chunk[len(layout)+1] = self.eos_token
        #print('chunk[len(layout)+1]: ',chunk[len(layout)+1])

        x = chunk[:-1]
        y = chunk[1:]
        return x,y


class MNISTLayout(MNIST):

    def __init__(self, root, train=True, download=True, threshold=32, max_length=None):
        super().__init__(root, train=train, download=download)
        self.vocab_size = 784 + 3  # bos, eos, pad tokens
        self.bos_token = self.vocab_size - 3
        self.eos_token = self.vocab_size - 2
        self.pad_token = self.vocab_size - 1

        self.threshold = threshold
        self.data = [self.img_to_set(img) for img in self.data]
        self.max_length = max_length
        if self.max_length is None:
            self.max_length = max([len(x) for x in self.data]) + 2  # bos, eos tokens
        self.transform = Padding(self.max_length, self.vocab_size)

    def __len__(self):
        return len(self.data)

    def img_to_set(self, img):
        fg_mask = img >= self.threshold
        fg_idx = fg_mask.nonzero(as_tuple=False)
        fg_idx = fg_idx[:, 0] * 28 + fg_idx[:, 1]
        return fg_idx

    def render(self, layout):
        layout = trim_tokens(layout, self.bos_token, self.eos_token, self.pad_token)
        x_coords = layout % 28
        y_coords = layout // 28
        # valid_idx = torch.where((y_coords < 28) & (y_coords >= 0))[0]
        img = np.zeros((28, 28, 3)).astype(np.uint8)
        img[y_coords, x_coords] = 255
        return Image.fromarray(img, 'RGB')

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) tokens from the data
        layout = self.transform(self.data[idx])
        return layout['x'], layout['y']

def rotate_xy(p, sin, cos, center):
    x_ = (p[:,0:1]-center[:,0:1])*cos-(p[:,1:2]-center[:,1:2])*sin+center[:,0:1]
    y_ = (p[:,0:1]-center[:,0:1])*sin+(p[:,1:2]-center[:,1:2])*cos+center[:,1:2]
#     print(((p[:,0:1]-center[:,0:1])*cos).shape, cos.shape, x_.shape)
    return np.hstack((x_, y_))


def draw_polygon_c(img,point, color):

    
    #print('point: ',point)  # draw box on canvas
    
    # cv2.line(img, tuple(point[1]), tuple(point[0]), color, 1)
    # cv2.line(img, tuple(point[1]), tuple(point[2]), color, 1)
    # cv2.line(img, tuple(point[2]), tuple(point[3]), color, 1)
    # cv2.line(img, tuple(point[0]), tuple(point[3]), color, 1)
    return img
class JSONLayout(Dataset):
    def __init__(self, np_path, max_length=None, precision=8):
        self.max_length = max_length
        self.categories = 1
        with open(np_path, "r") as f:
            self.data = np.load(np_path)
        '''num_paramaters: x,y,w,h(normalized from 0 to pi),angle(normalized from 0 to pi)'''
        data_length,num_box,num_paramaters = self.data.shape
        #print('self.data: ',self.data)
        self.data = self.data.reshape(data_length,num_box*num_paramaters)
        self.size = pow(2, precision)
        self.vocab_size = 320 + 3 #+ self.size
        self.bos_token = self.vocab_size - 3
        self.eos_token = self.vocab_size - 2
        self.pad_token = self.vocab_size - 1
        self.transform = Padding(self.max_length, self.vocab_size)
        self.colors = gen_colors(self.categories)

    def quantize_box(self, boxes, width, height):

        # range of xy is [0, large_side-1]
        # range of wh is [1, large_side]
        # bring xywh to [0, 1]
        boxes[:, [2, 3]] = boxes[:, [2, 3]] - 1
        boxes[:, [0, 2]] = boxes[:, [0, 2]] / (width - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]] / (height - 1)
        boxes = np.clip(boxes, 0, 1)

        # next take xywh to [0, size-1]
        boxes = (boxes * (self.size - 1)).round()

        return boxes.astype(np.int32)

    def __len__(self):
        return len(self.data)

    def render(self, layout,save_path,i):
        #print('layout: ',layout)
        # print('layout: ',layout)
        img = Image.new('RGB', (400, 400), color=(255, 255, 255))
        draw = ImageDraw.Draw(img, 'RGBA')
        layout = layout.reshape(-1)
        layout = trim_tokens(layout, self.bos_token, self.eos_token, self.pad_token)
        layout = layout[: len(layout) // 5 * 5].reshape(-1, 5)
        #print('layout: ',layout)
        #print('layout.shape: ',layout.shape)
        
        #teansfer data to normal size
        layout = layout.astype(np.float32)
        layout[:,0:4] = layout[:,0:4]/320.0
        # print('layout[:,0:4]: ',layout[:,0:4])
        layout[:,4:5] = (layout[:,4:5]*np.pi/320.0)-np.pi/2
        # print('layout: ',layout)
        
        #draw
        ld = np.hstack((layout[:,0:1]-layout[:,2:3]/2, layout[:,1:2]-layout[:,3:4]/2))
        rd = np.hstack((layout[:,0:1]+layout[:,2:3]/2, layout[:,1:2]-layout[:,3:4]/2))
        ru = np.hstack((layout[:,0:1]+layout[:,2:3]/2, layout[:,1:2]+layout[:,3:4]/2))
        lu = np.hstack((layout[:,0:1]-layout[:,2:3]/2, layout[:,1:2]+layout[:,3:4]/2))
        sinO = np.sin(layout[:,4:5])
        cosO = np.cos(layout[:,4:5])
        ld_r = rotate_xy(ld, sinO, cosO, layout[:,0:2])
        rd_r = rotate_xy(rd, sinO, cosO, layout[:,0:2])
        ru_r = rotate_xy(ru, sinO, cosO, layout[:,0:2])
        lu_r = rotate_xy(lu, sinO, cosO, layout[:,0:2])
        box_r = np.hstack((ld_r, rd_r, ru_r, lu_r)).reshape(len(layout[:,0:2]), -1, 2)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        #print('box_r.shape: ',box_r.shape)
        '''box_r.shape:  (32, 4, 2)'''
        plt.figure(figsize=(500,500),dpi=1)
        plt.axis('off')
        for box in box_r:
            #print('box: ',box)
            plt.plot(np.concatenate((box[:,0],box[0:1,0]),axis=0), np.concatenate((box[:,1],box[0:1,1]),axis=0), "b",linewidth=100)
        #print(save_path+f'/{i}.png')
        plt.savefig(save_path+f'/{i}.png', dpi=1)

        # img = np.ones((500, 500, 3),dtype=np.uint8)
        # img *= 255
        # for j, p in enumerate(box_r):
        #     if p[0][0]>=0:
        #         p = np.array(p).astype(int)
        #         #print(p)
        #         img = draw_polygon_c(img,p,(255,0,0))
        # cv2.imwrite(f'{dir_path}'+str(epoch).zfill(6)+'.jpg',img)
        
        # for i in range(len(layout)):
        #     x1, y1, x2, y2 = box_r[i]
        #     cat = layout[i][0]
        #     col = self.colors[cat-self.size] if 0 <= cat-self.size < len(self.colors) else [0, 0, 0]
        #     draw.rectangle([x1, y1, x2, y2],
        #                    outline=(0,0,255),#tuple(col) + (200,),
        #                    #fill=tuple(col) + (64,),
        #                    width=1)

        # # Add border around image
        # img = ImageOps.expand(img, border=2)
        
        # return img

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) tokens from the data
        layout = torch.tensor(self.data[idx])

        x, y = self.transform(layout)

        return x, y

if __name__ =='__main__':
    np_path = '/scratch/sg7484/data/InfiniteCityGen/origin/origin_test.npy'
    save_path = '/scratch/sg7484/InfiniteCityGen/results/test'
    train_dataset = JSONLayout(np_path,max_length = 161)
    loader = DataLoader(train_dataset, shuffle=True, pin_memory=True,
                    batch_size=1,
                    num_workers=0)
    for x,y in loader:
        # print(x)
        # print(y)
        y = y.detach().cpu().numpy()
        train_dataset.render(y,save_path,1)
        break