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
        query = (B, h, S2, d_q)
        key = (B, h, S1, d_k)
        value = (B, h, S1, d_v)
        mask = (B, Real_S, Real_S) or Triangular
        d_q = d_k = d_v
        '''
        batch_size = query.size(0)
        seq_size = value.size(2)
        d_k = query.size(2)

        #(B, h, S2, S1)
        att_score = torch.matmul(query / (d_k ** 0.5), key.transpose(2,3))

        if mask is not None:
            att_score = att_score.masked_fill(mask, -1e8)

        #print(att_score)
        #(B, h, S2, S1)
        att_score = F.softmax(att_score, dim = -1)
        #print(att_score[0])
        #(B, h, S2, d_v)
        output = torch.matmul(att_score, value)
        
        return output, att_score

        
class Multi_Head(nn.Module):
    def __init__(self, h, d_model, dropout):
        super(Multi_Head, self).__init__()
        self.h = h
        self.d_v = d_model // h
        self.d_m = d_model

        
        self.k_w = nn.Linear(self.d_m, self.d_m)
        self.q_w = nn.Linear(self.d_m, self.d_m)
        self.v_w = nn.Linear(self.d_m, self.d_m)

        self.layer_norm = nn.LayerNorm(self.d_m, eps = 1e-6)
        

        self.d_att = Attention()
        self.multi_head = nn.Linear(self.h * self.d_v, self.d_m)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask = None):
        '''
        x = (B, S, embed)
        maks = (B, S_real = True + padd = False)
        '''
        batch = k.shape[0]
        seq = k.shape[1]
        len_q, len_k, len_v = q.size(1), k.size(1), v.size(1)

        res = q
        #q = self.layer_norm(q)

        #B,S,d_m --> B, S ,h, d_v
        key = self.k_w(k).view(batch, len_k, self.h, self.d_v)
        query = self.q_w(q).view(batch, len_q, self.h, self.d_v)
        value = self.v_w(v).view(batch, len_v, self.h, self.d_v)
        
        key, query, value = key.transpose(1,2), query.transpose(1,2), value.transpose(1,2)
        
        #(B,S,S)
        if mask is not None:
            mask = mask.unsqueeze(1)

        output, att_score = self.d_att(query, key, value, mask)

        #B,h,S2,d_v --> B,S2,h*d_v
        output = output.transpose(1,2).contiguous().view(batch, len_q, -1)

        #B, S, d_model
        output = self.multi_head(output)
        output = self.layer_norm(res + output)
        output = self.dropout(output)

        return output, att_score