import torch
import torch.nn as nn
import torch.nn.functional as F
from model.sublayer import *

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_size, d_ff, dropout, max_len, h, Num, device):
        super(Encoder, self).__init__()

        self.vocab_size= vocab_size
        self.emb_size = emb_size

        self.device = device
        self.pos_enc = Pos_Encoding(vocab_size, emb_size, max_len, dropout, device)
        self.attention = nn.ModuleList([Encoding_layer(vocab_size, emb_size, d_ff, dropout, max_len, h) for _ in range(Num)])

    def forward(self, x, src_mask):
        '''
        x  = (B, S)
        '''
        #B, S, embed
        output = self.pos_enc(x)
        
        for layer in self.attention:
            output = layer(output, src_mask)

        return output

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_size, d_ff, dropout, max_len, h, Num, device):
        super(Decoder, self).__init__()

        self.vocab_size = vocab_size
        self.emb_size = emb_size

        self.pos_enc = Pos_Encoding(vocab_size, emb_size, max_len, dropout, device)
        self.attention = nn.ModuleList([Decoder_layer(vocab_size, emb_size, d_ff, dropout, max_len, h) for _ in range(Num)])

    def forward(self, x, src, src_mask, trg_mask):
        '''
        x  = (B, S_de)
        src = (B, S_en)
        src_mask = (B, S_r, S_r)
        trg_mask : triangular
        '''
        #B, S, embed
        output = self.pos_enc(x)
        
        for layer in self.attention:
            output = layer(output, src, trg_mask, src_mask)

        return output

class Transformer(nn.Module):
    def __init__(self, encoder, decoder, padd_idx, device):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.padd_idx = padd_idx
        self.device = device

        emb = self.encoder.emb_size
        out_vocab = self.decoder.vocab_size

        self.output = nn.Linear(emb, out_vocab)

    def gen_src_mask(self, x):
        '''
        x = (B, S)
        src_mask = (B, 1, S_r) --> broadcast
        '''
        #(B,1,S)
        src_mask = (x == self.padd_idx).unsqueeze(1)

        return src_mask.to(self.device)

    def gen_trg_mask(self, x):
        '''
        x = (B,S)
        trg_mask = (B, S, S_r) : triangle
        '''
        batch = x.shape[0]
        seq = x.shape[1]

        #B, 1, S
        trg_pad = (x == self.padd_idx).unsqueeze(1)
        #1, S, S
        #trg_idx = (torch.tril(torch.ones(seq,seq)) == 0).unsqueeze(0)
        #B, S, S
        trg_idx = (torch.tril(torch.ones(seq,seq)) == 0).repeat(batch, 1).view(batch, seq, seq).to(self.device)


        trg_mask = trg_pad | trg_idx

        return trg_mask

    def forward(self, src, trg):
        '''
        src = (B, S_source)
        trg = (B, S_target)
        '''
        batch = trg.shape[0]
        seq = trg.shape[1]

        src_mask = self.gen_src_mask(src)
        trg_mask = self.gen_trg_mask(trg)

        #print(src_mask.shape , trg_mask.shape)

        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, src_mask, trg_mask)


        output = output.view(batch * seq , -1)
        output = self.output(output)

        return output

    def inference(self,x):
        print(0)
        '''
        x  = (B, S_source)
        return (B, S_target)
        '''