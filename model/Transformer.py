import torch
from torch.nn import Transformer
import torch.nn as nn
from model.sublayer import Pos_encoding
import torch.nn.functional as F

class Transformer_fr(nn.Module):
    def __init__(self, en_vocab_size, de_vocab_size, padding_idx, max_len, embed_size, device):
        super(Transformer_fr, self).__init__()
        self.en_vocab = en_vocab_size
        self.de_vocab = de_vocab_size
        self.padd = padding_idx
        self.BOS = 1
        self.device = device

        self.encode = Pos_encoding(embed_size, max_len, device)
        self.en_emb = nn.Embedding(self.en_vocab, embed_size, padding_idx = 0)
        self.de_emb = nn.Embedding(self.de_vocab, embed_size, padding_idx = 0)

        self.transformer = Transformer()
        self.fc = nn.Linear(embed_size, self.de_vocab)

        self.scale = embed_size ** 0.5

    def forward(self, src, trg):

        src = self.en_emb(src) * self.scale + self.encode(src)
        trg = self.de_emb(trg) * self.scale+ self.encode(trg)

        src = src.transpose(0,1)
        trg = trg.transpose(0,1)

        output = self.transformer(src, trg)
        output = output.transpose(0,1)
        output = self.fc(output)

        #print(src.shape, trg.shape, output.shape)
        return output
    

    def inference(self, src, max_seq):
        '''
        x  = (B, S_source)
        return (B, S_target)
        '''

        #in order to paper, max_seq = src seq + 300
        max_seq = src.size(1)

        batch = src.size(0)
        output = torch.zeros((batch, max_seq)).to(torch.long).to(self.device)
        output[:, 0] = self.BOS

        for i in range(1,max_seq):            
            out = self.forward(src, output)

            #out = out.view(batch, max_seq, -1)

            out = out[:,i,:]
            _, pred = torch.topk(F.softmax(out), 1, dim = 1)
            output[:,i] = pred.squeeze(1)

        return output.detach()