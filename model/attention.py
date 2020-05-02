import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, dim , dropout = 0.1):
        super(Attention , self).__init__()
        self.dim = dim
        self.dropout = nn.Dropout(dropout)
        self.scale = (dim ** 0.5)

    def forward(self, query, key, value, mask = None):
        '''
        query = (B * h, S2, d_q)
        key = (B * h, S1, d_k)
        value = (B * h, S1, d_v)
        mask = (B, Real_S, Real_S) or Triangular
        d_q = d_k = d_v
        '''
        #(B * h, S2, S1)
        att_score = torch.bmm(query / self.scale, key.transpose(1,2))

        if mask is not None:
            att_score = att_score.masked_fill(mask, float("-inf"))

        #print(att_score)
        #(B *h, S2, S1)
        att_score = F.softmax(att_score, dim = -1)
        att_score = self.dropout(att_score)
        #print(att_score[0])
        #(B *h, S2, d_v)
        output = torch.bmm(att_score, value)
        
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

        self.layer_norm = nn.LayerNorm(self.d_m)
        

        self.d_att = Attention(self.d_v)
        self.multi_head = nn.Linear(self.h * self.d_v, self.d_m)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask = None):
        '''
        x = (B, S, embed)
        maks = (B, S_real = True + padd = False)
        '''
        batch = k.shape[0]
        len_q, len_k, len_v = q.size(1), k.size(1), v.size(1)

        res = q
        #q = self.layer_norm(q)

        #B,S,d_m --> B, S ,h, d_v
        key = self.k_w(k).view(batch, len_k, self.h, self.d_v)
        query = self.q_w(q).view(batch, len_q, self.h, self.d_v)
        value = self.v_w(v).view(batch, len_v, self.h, self.d_v)
        
        key, query, value = key.transpose(1,2).contiguous(), query.transpose(1,2).contiguous(), value.transpose(1,2).contiguous()
        key, query, value = key.view(-1, len_k, self.d_v), query.view(-1, len_q, self.d_v), value.view(-1, len_v, self.d_v)

        '''
        #(B,S,S)
        if mask is not None:
            mask = mask.unsqueeze(1)
        '''
        output, att_score = self.d_att(query, key, value, mask)

        #B * h,S2,d_v --> B,S2,h*d_v
        output = output.view(batch, self.h, len_q, self.d_v)
        output = output.transpose(1,2).contiguous().view(batch, len_q, -1)

        #B, S, d_model
        output = self.multi_head(output)
        output = self.dropout(output)
        output = self.layer_norm(res + output)

        return output, att_score