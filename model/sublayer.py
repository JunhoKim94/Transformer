import torch
import torch.nn as nn
from model.attention import *
import torch.nn.functional as F

class Pos_Encoding(nn.Module):
    def __init__(self, vocab_size, emb_size, max_length, dropout):
        super(Pos_Encoding, self).__init__()
        '''
        PE(pos, 2i) = sin(pos/ 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos/ 10000^(2i/d_model))
        i --> d_model
        '''
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.max_length = max_length

        self.embed = nn.Embedding(vocab_size, emb_size)
        self.pos_emb = nn.Embedding(max_length, emb_size)

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([emb_size])).cuda()

    def initialize(self):
        for i in range(self.max_length):
            for j in range(self.emb_size):
                if j % 2 == 0:
                    self.pos_emb.weight.data[i, j] = torch.sin(torch.Tensor([i / (10000**(2 * j / 10))]))
                else:
                    self.pos_emb.weight.data[i, j] = torch.cos(torch.Tensor([i / (10000**(2 * j / 10))]))
        for params in self.pos_emb.parameters():
            params.requires_grad = False

    def forward(self, x):
        '''
        x = (B, B_S)
        '''
        batch= x.shape[0]
        seq = x.shape[1]

        #B, B_S
        pos = torch.arange(0, seq).unsqueeze(0).repeat(batch,1).cuda()
        #B, B_S, embed
        output = self.pos_emb(pos) / self.scale + self.embed(x)
        output = self.dropout(output)

        return output

class Position_wise_FFN(nn.Module):
    def __init__(self, emb_size, d_ff, dropout):
        super(Position_wise_FFN, self).__init__()
        self.emb_size = emb_size
        self.d_ff = d_ff

        self.linear = nn.Sequential(
            nn.Linear(emb_size, self.d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_ff, emb_size)
        )
    
    def forward(self, x):
        '''
        x = (B, S, embed_size = d_model)
        '''
        return self.linear(x)


class Encoding_layer(nn.Module):
    def __init__(self, vocab_size, emb_size, d_ff, dropout, max_len, h):
        super(Encoding_layer, self).__init__()
        self.vocab_size = vocab_size
        self.emb_size =emb_size
        self.d_ff = d_ff
        self.max_len = max_len

        #self.pos_enc = Pos_Encoding(vocab_size, emb_size, max_len, dropout)
        #self.layer_norm = nn.LayerNorm(emb_size)
        self.multi_head =  Multi_Head(h, emb_size, dropout)
        self.fnn = Position_wise_FFN(emb_size, d_ff, dropout)

    def forward(self, x, src_mask):
        '''
        x = (B, S, embed)
        out = (B, S, embed)
        src_mask = (B, S_r, S_r)
        '''

        #B,S,embed
        output = self.multi_head(x, x ,x, src_mask)

        #B,S,embed
        output = F.layer_norm(x + output)

        res = output
        #B,S,embed
        output = self.fnn(output)
        output = F.layer_norm(res + output)

        return output

class Decoder_layer(nn.Module):
    def __init__(self, vocab_size, emb_size, d_ff, dropout, max_len, h):
        super(Decoder_layer, self).__init__()
        self.vocab_size = vocab_size
        self.emb_size =emb_size
        self.d_ff = d_ff
        self.max_len = max_len
        
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
        output = self.multi_head(x, x, x, trg_mask)
        #B,S,embed
        output = F.layer_norm(x + output)
       
        res = output
        #B,S,embed --> encoder들 (src)의 attention 이므로 srg_mask
        output = self.encoder_head(src, src, output, src_mask)
        output = F.layer_norm(res + output)

        res = output
        #B,S,embed
        output = self.fnn(output)
        output = F.layer_norm(res + output)

        return output