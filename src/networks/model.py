'''
This file contains the architecture of the weighted shapley estimator
References
- https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html
- https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
'''


import torch
import torch.nn as nn
import sys
import numpy as np
from shapley.models.resnet import BasicBlock, conv3x3, FitModule
import torch.nn.functional as F
from shapley.utils.shap_utils import return_model
import os
from torch import linalg as LA
from .losses import CrossSupConLoss, SelfSupConLoss

def attention(q, k, v, d_k, dropout=None, is_cosine=False):
    
    if not is_cosine:
        attention_matrix = torch.matmul(q, k.T)
        scores = F.softmax(attention_matrix/ (d_k ** 0.5), dim=-1)
    else:
        q_norm = LA.vector_norm(q, ord=2, dim = -1).reshape(-1,1).repeat(1,q.shape[-1])
        q_scaled = q / q_norm
        q_scaled = torch.nan_to_num(q_scaled, nan=1e-6)

        k_norm = LA.vector_norm(k, ord=2, dim = -1).reshape(-1,1).repeat(1,k.shape[-1])
        k_scaled = k / k_norm
        k_scaled = torch.nan_to_num(k_scaled, nan=1e-6)

        # scores = (1 + torch.matmul(q_scaled, k_scaled.T))/2
        attention_matrix = torch.matmul(q_scaled, k_scaled.T)
        scores = F.relu(attention_matrix) # much better than random

    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v)
    return attention_matrix, output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=256, dropout = 0.1):
        super().__init__() 
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, v_out_dim, dropout = 0.1, is_cosine = False):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.is_cosine = is_cosine
        
        self.x_q_linear = nn.Linear(d_model, d_model)
        self.x_v_linear = nn.Linear(d_model, v_out_dim)
        self.x_k_linear = nn.Linear(d_model, d_model)

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, v_out_dim)
        self.k_linear = nn.Linear(d_model, d_model)

        self.d_k = d_model
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(v_out_dim, v_out_dim)
        self.x_sup_con_loss = CrossSupConLoss()
        self.self_sup_con_loss = SelfSupConLoss()
        
    def forward(self, x1, x2, labels_y1, labels_y2):
        
        # Self attention
        k = self.k_linear(x1)    #.view(bs,  self.h, self.d_k)
        q = self.q_linear(x1)
        # v = self.v_linear(v)

        attention_matrix = torch.matmul(q, k.T) / (self.d_k ** 0.5)
        # self_con_loss = self.self_sup_con_loss(features=attention_matrix, labels=labels_y1)

        # Cross attention
        x_k = self.x_k_linear(x2)    #.view(bs,  self.h, self.d_k)
        x_q = self.x_q_linear(x1)
        x_v = self.x_v_linear(x2)
        
        x_attention_matrix = torch.matmul(x_q, x_k.T) / (self.d_k ** 0.5)
        masked_prob, x_con_loss = self.x_sup_con_loss(features=x_attention_matrix, labels_y1=labels_y1, labels_y2=labels_y2)
    
        # clamp min prob in mask_prob
        # masked_prob = torch.clamp(masked_prob, min=1e-3, max=1)
        loss = x_con_loss # + self_con_loss
        # if torch.isnan(loss):
        #     print("error")
        output = torch.matmul(masked_prob, x_v)
        output = F.normalize(x1 + output)
        
        output = F.relu(self.out(output))
    
        return loss, output

class Encoder(nn.Module):
    def __init__(self, dim_in, feat_dim) -> None:
        super(Encoder, self).__init__()
        self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )

    def forward(self, x):
        return self.head(x)
        
# takes two inputs of length 'input_dim' each and gives a single scalar as output
class EstimatorNetwork(nn.Module):
    def __init__(self, heads=1, d_model=512, l_embeddings = 64, ff_dim = 512, dropout = 0.1):
        super(EstimatorNetwork, self).__init__()

        # self.norm_x1 = torch.nn.LayerNorm(d_model)
        # self.norm_x2 = torch.nn.LayerNorm(d_model)

        self.enc_x1 = Encoder(512, d_model)
        self.enc_x2 = Encoder(512, d_model)
        
        self.attn = MultiHeadAttention(heads, d_model, d_model, dropout, is_cosine=False)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, ff_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ff_dim, 1)
        ) # FeedForward(d_model, ff_dim)


    def forward(self, x1,y1, x2, y2):
        
        enc_x1 = self.enc_x1(F.normalize(x1))
        enc_x2 = self.enc_x2(F.normalize(x2))

        loss, output = self.attn(enc_x1,enc_x2, y1,y2)
        result = self.ff(output)
        
        if torch.isnan(torch.sum(result, dim=1)[0]):
            print("Nan occured")
        return loss, result

if __name__=='__main__':
    # unit test
    d = 512
    model = EstimatorNetwork(d_model=d
                            ,ff_dim = 256)
    x1 = torch.randn(5, 11, 28, 28)      # train
    x2 = torch.randn(15, 11, 28, 28)    # test
    output = model(x1, x2)
    print(output)
    print(output.shape)
