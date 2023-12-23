import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn



class CityPolyCrit(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, remain_num = 6, max_poly = 20, max_build = 60, device = 'cuda'):
        super().__init__()
        # --------------------------------------------------------------------------
        self.len_criterion = nn.CrossEntropyLoss()
        self.remain_num = remain_num
        self.max_poly = max_poly
        self.max_build = max_build
        self.device = device

        # --------------------------------------------------------------------------

    
    def forward(self, pred, len_tar, poly_tar):
        bsz, _, _ = pred.shape
        pred_poly = pred[:,self.remain_num:,:self.max_poly*2].view(bsz, -1, self.max_poly, 2)
        pred_len = pred[:,self.remain_num:,self.max_poly*2:]

        mask = torch.zeros([bsz, self.max_build-self.remain_num, self.max_poly])
        for i in range(len(len_tar)):
            for j in range(len(len_tar[i])):
                mask[i, j, :len_tar[i][j]] = 1

        loss_l1 = torch.mean(torch.masked_select(torch.abs(pred_poly-poly_tar), 
                                                     mask.unsqueeze(-1).repeat(1, 1, 1, 2).bool().to(self.device)))

        loss_len = self.len_criterion(torch.cat([pred_len[i, :len(len_tar[i]), :] for i in range(len(len_tar))], dim = 0), 
                                     (torch.cat(len_tar, dim = 0)-1).long().to(self.device))

        return loss_l1, loss_len