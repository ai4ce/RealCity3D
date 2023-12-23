import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
import time

from vertex_embed import VertexEmbed
from model_autoregress import AutoModel


class ModelFace(nn.Module):
    def __init__(self, latent_dim = 256, padding_idx = 1000, num_layer = 4, device = 'cuda'):
        super().__init__()
        self.latent_dim = latent_dim
        self.device = device
        self.padding_idx = padding_idx

        self.vertices_encoder = VertexEmbed(ouput_dim=latent_dim, device = device)

        self.start_token = nn.Parameter(torch.zeros(1, 1, self.latent_dim))

        self.automodel = AutoModel(padding_idx = 0, embed_dim = latent_dim, num_layer = num_layer, device=device)

        self.loss_crit = nn.CrossEntropyLoss(ignore_index=padding_idx)

        torch.nn.init.normal_(self.start_token, std=.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    
    
    def forward(self, source_v, length_v, source_f, length_f, target):
        bsz, f_len = source_f.shape

        vertex_encoding = self.vertices_encoder(source_v)
        auto_in = torch.zeros([bsz, f_len, self.latent_dim]).to(self.device)
        length_f_in = length_f-1
        for i in range(bsz):
            auto_in[i, :length_f_in[i], :] = torch.gather(vertex_encoding[i], dim = 0, index=source_f[i, :length_f_in[i]].unsqueeze(-1).repeat(1, self.latent_dim))

        out = self.automodel(torch.cat((self.start_token.repeat(bsz, 1, 1), auto_in), dim = 1))

        loss = 0
        logitout = []
        for i in range(bsz):
            logits = torch.matmul(out[i,:length_f[i]], vertex_encoding[i, :2+length_v[i]].permute(1, 0))
            logitout.append(logits)
            # assert logits.shape[1]>max(target[i,:length_f[i]])
            # assert 0<=min(target[i,:length_f[i]])
            loss += self.loss_crit(logits, target[i,:length_f[i]])
            
        loss = loss/bsz

        return loss, logitout
    
    
    def generate(self, file_path, vertices, firstface):
        assert vertices.shape[0] == 1
        #print(vertices.shape[1])
        out = [i_d+1 for i_d in firstface]+[0]
        file = open(file_path, "w")
        for vertex in vertices[0]:
            ver = "v "+f"{vertex[0]}"+" "+f"{vertex[1]}"+" "+f"{vertex[2]}"+" "
            file.write(ver)
            file.write("\n")
        # file.write("f "+f"{vertex[0]}"+" "+f"{vertex[1]}"+" "+f"{vertex[2]}"+" ")
        
        firstVertex = -1
        listVertices = []
        IfBegin = False
        unrefVertices = set()
        for i in range(2+vertices.shape[1]):
            unrefVertices.add(i)
        unrefVertices.remove(0)
        unrefVertices.remove(1)
        # print(unrefVertices)

        while True:
            _, pred = self(vertices, torch.tensor([[vertices.shape[1]]]).to(self.device), 
                           torch.tensor(out).unsqueeze(0).to(self.device), 
                           torch.tensor([[len(out)+1]]).to(self.device), 
                           torch.tensor(out+[0]).unsqueeze(0).to(self.device))
            pred_prob = pred[0][-1].softmax(-1)
            #print(pred_prob.shape) # [num_vertices +2, ]
            # idx = -1

            while True:
                idx = torch.argmax(pred_prob)
                if IfBegin == False:
                    IfBegin = True
                    firstVertex = idx
                if len(out) <= 1:
                    #print(22)
                    break
                if len(out)==0 or out[-1] == 0:
                    listVertices = []
                if out[-1] == 0 and idx == 1:
                    break
                if out[-1] == 0 and idx == 0:
                    pred_prob[idx] = 0
                    continue
                if out[-1] == 0:
                    if idx < firstVertex:
                        pred_prob[idx] = 0
                        continue
                    if len(unrefVertices) > 0 and idx > min(unrefVertices):
                        pred_prob[idx] = 0
                        continue
                    firstVertex = idx
                else:
                    if idx == 0:
                        break
                    if idx <= firstVertex:
                        pred_prob[idx] = 0
                        continue
                    if idx in listVertices:
                        pred_prob[idx] = 0
                        continue
                break
            
            out.append(idx)
            listVertices.append(idx)
            if idx.item() in unrefVertices:
                unrefVertices.remove(idx.item())
            #print(idx.item())
            if idx == 1:
                break
        
        face = "f "
        for id in out:
            if id == 1:
                break
            elif id == 0:
                file.write(face)
                file.write("\n")
                face = "f "
            else:
                face += f"{id-1}"+" "
        return out
    

    # def generate(self, file_path, vertices):
    #     assert vertices.shape[0] == 1
    #     print(vertices.shape[1])
    #     out = []
    #     file = open(file_path, "w")
    #     for vertex in vertices[0]:
    #         ver = "v "+f"{vertex[0]}"+" "+f"{vertex[1]}"+" "+f"{vertex[2]}"+" "
    #         file.write(ver)
    #         file.write("\n")
    #     while True:
    #         _, pred = self(vertices, torch.tensor([[vertices.shape[1]]]).to(self.device), 
    #                        torch.tensor(out).unsqueeze(0).to(self.device), 
    #                        torch.tensor([[len(out)+1]]).to(self.device), 
    #                        torch.tensor(out+[0]).unsqueeze(0).to(self.device))
    #         pred_prob = pred[0][-1].softmax(-1)
    #         idx = torch.argmax(pred_prob)
    #         out.append(idx)
    #         if idx == 1:
    #             break
        
    #     face = "f "
    #     for id in out:
    #         if id == 1:
    #             break
    #         elif id == 0:
    #             file.write(face)
    #             file.write("\n")
    #             face = "f "
    #         else:
    #             face += f"{id-1}"+" "

    #     return out