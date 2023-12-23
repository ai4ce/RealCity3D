import numpy as np
import cv2
import os
from shapely.geometry import Polygon
import json
import torch

def box2corners_th(box):
        """convert box coordinate to corners

        Args:
            box (torch.Tensor): (B, N, 5) with x, y, w, h, alpha

        Returns:
            torch.Tensor: (B, N, 4, 2) corners
        """
        B = box.size()[0]
        x = box[..., 0:1]
        y = box[..., 1:2]
        w = box[..., 2:3]
        h = box[..., 3:4]
        alpha = box[..., 4:5] # (B, N, 1)
        # print('cos: ',cos)
        x4 = torch.FloatTensor([0.5, -0.5, -0.5, 0.5]).unsqueeze(0).unsqueeze(0).to(box.device) # (1,1,4)
        x4 = x4 * w     # (B, N, 4)
        y4 = torch.FloatTensor([0.5, 0.5, -0.5, -0.5]).unsqueeze(0).unsqueeze(0).to(box.device)
        y4 = y4 * h     # (B, N, 4)
        corners = torch.stack([x4, y4], dim=-1)     # (B, N, 4, 2)
        sin = torch.sin(alpha)
        cos = torch.cos(alpha) 
        row1 = torch.cat([cos, sin], dim=-1)
        row2 = torch.cat([-sin, cos], dim=-1)       # (B, N, 2)
        rot_T = torch.stack([row1, row2], dim=-2)   # (B, N, 2, 2)
        rotated = torch.bmm(corners.view([-1,4,2]), rot_T.view([-1,2,2]))
        rotated = rotated.view([B,-1,4,2])          # (B*N, 4, 2) -> (B, N, 4, 2)
        rotated[..., 0] += x
        rotated[..., 1] += y
        return rotated

if __name__=="__main__":
    data = np.load('../../datasets/boxstates/states_xy.npy')
    
    boxes = np.array(box2corners_th(torch.tensor(data[:1000])))*500

    list_area=[]

    for i, box in enumerate(boxes):
        img = np.ones((500,500,3),np.uint8)*255
        gen_poly = box

        for j in range(len(gen_poly)):
            poly = gen_poly[j]
            pts = np.array(poly, np.int32)
            pts = pts.reshape((-1,1,2)).astype(int)
            list_area.append(Polygon(poly).area)

            cv2.fillPoly(img, [pts], color=(255, 255, 0))
            cv2.polylines(img,[pts],True,(0,0,0),1)
    
        dir_path = '../../results/test/box/gt_image_box/img/'
        if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        print(i)
                
        cv2.imwrite(f'{dir_path}'+str(i) +'.jpg',img)

    dict_all = {'area': list_area}

    with open(f'../../results/test/box/gt_image_box/'+'wd.json', 'w') as json_file:
        json.dump(dict_all, json_file)