import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, mode = "dot", dim = None):
        super(Attention , self).__init__()
        self.mode = mode
        self.dim = dim

    def forward(self, query, key, value, mask = None):
        '''
        query = (B, S2, d_q)
        key = (B, S1, d_k)
        value = (B, S1, d_v)
        mask = (B, Real_S, Real_S) or Triangular
        d_q = d_k = d_v
        '''
        batch_size = query.size(0)
        seq_size = value.size(1)
        d_k = query.size(2)

        #(B, S2, S1)
        att_score = torch.bmm(query, key.transpose(1,2)) / (d_k **0.5)

        if mask is not None:
            att_score.mask_fill_(mask, -1e10)

        att_score = F.softmax(att_score.view(-1, seq_size), dim = 1).view(batch_size, -1, seq_size)

        #(B, S2, d_v)
        output = torch.bmm(att_score, value)
        
        return output

        
class Multi_Head(nn.Module):
    def __init__(self, h, d_model, dropout):
        super(Multi_Head, self).__init__()
        self.h = h
        self.d_v = d_model // h
        self.d_m = d_model
        '''
        self.k_w = nn.Linear(self.d_m, self.d_m)
        self.q_w = nn.Linear(self.d_m, self.d_m)
        self.v_w = nn.Linear(self.d_m, self.d_m)

        '''
        self.k_w = nn.ModuleList([nn.Linear(self.d_m, self.d_v) for _ in range(h)])
        self.q_w = nn.ModuleList([nn.Linear(self.d_m, self.d_v) for _ in range(h)])
        self.v_w = nn.ModuleList([nn.Linear(self.d_m, self.d_v) for _ in range(h)])
        
        self.d_att = nn.ModuleList([Attention() for i in range(h)])
        self.multi_head = nn.Linear(self.h * self.d_v, self.d_m)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask = None):
        '''
        x = (B, S, embed)
        maks = (B, S_real = True + padd = False)
        '''
        batch = k.shape[0]
        seq = k.shape[1]

        #B,S,d_v
        key = [f(k) for f in self.k_w]
        query = [f(q) for f in self.q_w]
        value = [f(v) for f in self.v_w]
        '''
        #B,S,d_m
        key = self.k_w(x)
        query = self.q_w(x)
        value = self.v_w(x)
        '''
        output = [att(query[i], key[i], value[i]) for i, att in enumerate(self.d_att)]
        
        #B, S, d_v * h
        output = torch.cat(output, 2)
        #B, S, d_model
        output = self.multi_head(output)

        return output