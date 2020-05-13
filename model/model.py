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
        self.pos_enc = Encoding(vocab_size, emb_size, max_len, dropout, device)
        self.attention = nn.ModuleList([Encoding_layer(emb_size, d_ff, dropout, h) for _ in range(Num)])
        self.layer_norm = nn.LayerNorm(emb_size)


    def forward(self, x, src_mask):
        '''
        x  = (B, S)
        '''
        #B, S, embed
        output = self.pos_enc(x)
        
        for layer in self.attention:
            output = layer(output, src_mask)

        output = self.layer_norm(output)

        return output


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_size, d_ff, dropout, max_len, h, Num, device):
        super(Decoder, self).__init__()

        self.vocab_size = vocab_size
        self.emb_size = emb_size

        self.pos_enc = Encoding(vocab_size, emb_size, max_len, dropout, device)
        self.attention = nn.ModuleList([Decoder_layer(emb_size, d_ff, dropout, h) for _ in range(Num)])
        self.layer_norm = nn.LayerNorm(emb_size)

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

        output = self.layer_norm(output)

        return output

class Transformer(nn.Module):
    def __init__(self, encoder, decoder, padd_idx, device):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.padd_idx = padd_idx
        self.device = device

        self.BOS = 1
        self.EOS = 2

        self.emb = self.encoder.emb_size
        self.out_vocab = self.decoder.vocab_size

        self.output = nn.Linear(self.emb, self.out_vocab)

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
        #trg_pad = (x == self.padd_idx).unsqueeze(1)
        
        #1, S, S
        trg_mask = (torch.tril(torch.ones(seq,seq)) == 0).unsqueeze(0).to(self.device)
        #B, S, S
        
        #trg_mask = (torch.tril(torch.ones(seq,seq)) == 0).repeat(batch, 1).view(batch, seq, seq).to(self.device)
        #trg_mask = trg_pad | trg_idx
        
        return trg_mask

    def forward(self, src, trg):
        '''
        src = (B, S_source)
        trg = (B, S_target)
        '''
        batch = trg.shape[0]
        seq = trg.shape[1]

        #src_mask = self.gen_src_mask(src)
        src_mask = None
        trg_mask = self.gen_trg_mask(trg)

        #print(src_mask.shape , trg_mask.shape)

        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, src_mask, trg_mask)

        #B, S, emb
        #output = output.view(batch * seq , -1)
        #output = output.view(-1, self.out_vocab)
        output = self.output(output)

        return output


    def inference(self, src):
        '''
        x  = (B, S_source)
        return (B, S_target)
        '''

        #in order to paper, max_seq = src seq + 300
        max_seq = src.size(1)
        batch = src.size(0)

        lengths = np.array([max_seq] * batch)
        #outputs = []

        outputs = torch.zeros((batch, 1)).to(torch.long).to(self.device)
        outputs[:, 0] = self.BOS

        for step in range(1,max_seq):            
            out = self.forward(src, outputs)

            #out = out.view(batch, max_seq, -1)
            #print(out.shape)
            out = out[:,-1,:]
            pred = torch.topk(F.log_softmax(out), 1, dim = -1)[1]

            outputs = torch.cat([outputs, pred], dim = 1)
            eos_batches = pred.data.eq(self.EOS)

            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = step


        return outputs.detach(), lengths