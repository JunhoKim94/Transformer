import torch
from torch.nn import Transformer
import torch.nn as nn
from model.sublayer import Pos_encoding,Encoding
import torch.nn.functional as F
import numpy as np

class Transformer_fr(nn.Module):
    def __init__(self, en_vocab_size, de_vocab_size, padding_idx, max_len, embed_size, device):
        super(Transformer_fr, self).__init__()
        self.en_vocab = en_vocab_size
        self.de_vocab = de_vocab_size
        self.padd = padding_idx
        self.BOS = 1
        self.EOS = 2
        self.device = device

        #self.encode = Pos_encoding(embed_size, max_len, device)
        #self.en_emb = nn.Embedding(self.en_vocab, embed_size, padding_idx = 0)
        #self.de_emb = nn.Embedding(self.de_vocab, embed_size, padding_idx = 0)

        self.en_enc = Encoding(self.en_vocab, embed_size, max_len, 0.2, device)
        self.de_enc = Encoding(self.de_vocab, embed_size, max_len, 0.2, device)

        self.transformer = Transformer()
        self.fc = nn.Linear(embed_size, self.de_vocab)

        self.scale = embed_size ** 0.5

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
        #trg_pad = (x == self.padd).unsqueeze(1)
        #1, S, S
        #S, S
        trg_mask = torch.tril(torch.ones(seq,seq))
        trg_mask[trg_mask == 0] = float("-inf")
        trg_mask[trg_mask == 1] = float(0.0)
        #trg_mask = trg_pad | trg_idx
        #print(trg_mask)

        return trg_mask.to(self.device)


    def forward(self, src, trg):

        #src = self.en_emb(src) * self.scale + self.encode(src)
        #trg = self.de_emb(trg) * self.scale+ self.encode(trg)
        trg_seq = trg.size(1)

        src = self.en_enc(src)
        trg = self.de_enc(trg)

        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq).to(self.device)
        #trg_mask = self.gen_trg_mask(trg)

        #print(trg_mask)
        src = src.transpose(0,1)
        trg = trg.transpose(0,1)

        output = self.transformer(src, trg, tgt_mask = trg_mask)
        output = output.transpose(0,1)
        output = self.fc(output)

        #print(src.shape, trg.shape, output.shape)
        return output
    

    def inference(self, src):
        '''
        x  = (B, S_source)
        return (B, S_target)
        '''

        #in order to paper, max_seq = src seq + 300
        max_seq = src.size(1) + 50
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