import torch
import torch.nn as nn
from model.attention import *
import torch.nn.functional as F
import math

class Pos_encoding(nn.Module):
    def __init__(self, emb_size, max_len, device):
        super(Pos_encoding, self).__init__()
        self.emb_size = emb_size
        self.max_len = max_len
        self.device = device

        #(Max_length, embed_size)
        pe = torch.zeros(self.max_len, self.emb_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2).float() * (-math.log(10000.0) / self.emb_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos_emb", pe)

    def forward(self,x):
        '''
        x = B,S
        '''
        seq_size = x.size(1)
        #S, embed_size
        #1,S,embed_size
        return self.pos_emb[:seq_size, :].unsqueeze(0)



class Encoding(nn.Module):
    def __init__(self, vocab_size, emb_size, max_length, dropout, device):
        super(Encoding, self).__init__()
        '''
        PE(pos, 2i) = sin(pos/ 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos/ 10000^(2i/d_model))
        i --> d_model
        '''
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.max_length = max_length
        self.device = device

        self.embed = nn.Embedding(vocab_size, emb_size, padding_idx = 0)
        self.pos_emb = Pos_encoding(emb_size, max_length, self.device)
        self.dropout = nn.Dropout(dropout)
        #self.scale = torch.sqrt(torch.FloatTensor([emb_size])).to(self.device)

    def forward(self, x):
        '''
        x = (B, B_S)
        '''
        batch= x.shape[0]
        seq = x.shape[1]

        #B, B_S, embed
        output = self.pos_emb(x) + self.embed(x) * math.sqrt(self.emb_size)
        output = self.dropout(output)
        
        return output

class Position_wise_FFN(nn.Module):
    def __init__(self, emb_size, d_ff, dropout):
        super(Position_wise_FFN, self).__init__()
        self.emb_size = emb_size
        self.d_ff = d_ff

        self.layer_norm = nn.LayerNorm(emb_size, eps = 1e-6)

        self.linear = nn.Sequential(
            nn.Linear(emb_size, self.d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_ff, emb_size),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        '''
        x = (B, S, embed_size = d_model)
        '''
        res = x
        out = self.linear(x)
        out = self.layer_norm(out + res)

        return out


class Encoding_layer(nn.Module):
    def __init__(self, emb_size, d_ff, dropout, h):
        super(Encoding_layer, self).__init__()
        self.emb_size = emb_size
        self.d_ff = d_ff

        #self.pos_enc = Pos_Encoding(vocab_size, emb_size, max_len, dropout)
        self.multi_head =  Multi_Head(h, emb_size, dropout)
        self.fnn = Position_wise_FFN(emb_size, d_ff, dropout)

    def forward(self, x, src_mask):
        '''
        x = (B, S, embed)
        out = (B, S, embed)
        src_mask = (B, S_r, S_r)
        '''
        #B,S,embed
        output, att_score = self.multi_head(x, x ,x, src_mask)

        #B,S,embed
        output = self.fnn(output)

        return output

class Decoder_layer(nn.Module):
    def __init__(self, emb_size, d_ff, dropout, h):
        super(Decoder_layer, self).__init__()
        self.emb_size = emb_size
        self.d_ff = d_ff
        self.h = h
        
        self.multi_head =  Multi_Head(h,  emb_size, dropout)
        self.encoder_head = Multi_Head(h,  emb_size, dropout)
        self.fnn = Position_wise_FFN(emb_size, d_ff, dropout)

    def forward(self, x, src, trg_mask, src_mask):
        '''
        x = (B, S, embed)
        out = (B, S, embed)
        trg = triangular
        '''

        #B,S,embed
        output, self_att = self.multi_head(x, x, x, trg_mask)
        #query = decoder, key & value = encoder
        #B,S,embed --> encoder들 (src)의 attention 이므로 src_mask
        output, co_att = self.encoder_head(output, src, src, src_mask)
        #B,S,embed
        output = self.fnn(output)

        return output