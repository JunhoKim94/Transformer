import numpy as np
import torch
import torch.nn as nn
import pickle
from preprocess import *
from model.model import Encoder, Decoder, Transformer
import time
import matplotlib.pyplot as plt
from utils import *
from loader import *
from model.Transformer import Transformer_fr

PAD = 0
lr = 0.0025
steps = 1000000
warm_up = 4000
bpe = True

print("\n ==============================> Training Start <=============================")
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
#device = torch.device("cpu")
print(torch.cuda.is_available())
corpus_path = "./corpus.pickle"
data_path = "./data/split/test.pickle"

en_word2idx, en_idx2word, de_word2idx, de_idx2word = call_data(corpus_path)

test_dict = dict()

en_word2idx["</w>"] = len(en_word2idx)
en_idx2word[len(en_idx2word)] = "</w>"
de_word2idx["</w>"] = len(de_word2idx)
de_idx2word[len(de_idx2word)] = "</w>"

print(len(en_word2idx) , len(de_word2idx))
print(len(en_idx2word) , len(de_idx2word))

en_vocab_size = len(en_word2idx)
de_vocab_size = len(de_word2idx)
emb_size = 512
d_ff = 2048
dropout = 0.1
max_len = 200
h = 8
Num = 6
max_token = 4096

#dataset
dataset = Basedataset(data_path, en_word2idx, de_word2idx, bpe)
dataloader = Batch_loader(dataset, device, max_len, max_token)

total = len(dataset)

#model
'''
encoder = Encoder(en_vocab_size, emb_size , d_ff, dropout, max_len, h, Num, device)
decoder = Decoder(de_vocab_size, emb_size , d_ff, dropout, max_len, h, Num, device)
model = Transformer(encoder, decoder, PAD, device).to(device)
'''

model = Transformer_fr(en_vocab_size, de_vocab_size, PAD, max_len, emb_size, device).to(device)

#model.load_state_dict(torch.load("./model.pt", map_location= device))
criterion = nn.CrossEntropyLoss()#(ignore_index = 0)
#criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = lr, betas = (0.9, 0.98), eps = 1e-9)

test_dataset = Basedataset("./data/split/test.pickle", en_word2idx, de_word2idx, bpe)
test_loader = Batch_loader(test_dataset, device, max_len, 2000)

#val_loader = Batch_loader(dataset, device, max_len, 2000)

best_loss = 1e5 #3.8261873620196707
st = time.time()

step_loss = 0
avg_batch = 0
avg_src_seq = 0
avg_trg_seq = 0
start = 0


for step in range(start + 1, start + steps+1):
    model.train()

    for param_group in optimizer.param_groups:
        param_group['lr'] =  emb_size**(-0.5) * min(step**(-0.5), step * (warm_up**(-1.5))) / 8
        lr = param_group['lr']

    #print(step)
    #sr_batch, tr_batch = test_loader.get_batch()
    sr_batch, tr_batch = dataloader.get_batch()
    #sr_batch, tr_batch = dataloader.get_normal_batch(64)

    target = tr_batch[:,1:]
    tr_batch = tr_batch[:,:-1]

    target = target.contiguous().view(-1)

    y_pred = model(sr_batch, tr_batch)

    y_pred = y_pred.view(-1, len(de_word2idx))

    #loss = cal_loss(y_pred, target, 0, True)
    loss = criterion(y_pred, target)

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 9, norm_type = 2)

    optimizer.step()
    step_loss += loss.item()
    avg_batch += len(sr_batch)
    avg_src_seq += sr_batch.size(1)
    avg_trg_seq += tr_batch.size(1)

    if step % 2000 == 0:
        d = step + 1e-5 - start
        spend = round((time.time() - st)/ 3600 ,3)
        print(f"total step : {steps + start}  |  curr_step : {step}  |  Time Spend : {spend} hours  | loss :  { step_loss / d} | lr : {lr} | avg_batch : {int(avg_batch / d)} | avg_src_seq : {int(avg_src_seq / d)} | avg_trg_seq : {int(avg_trg_seq / d)}")
        torch.save(model.state_dict(), "./current_step.pt")
        #score = evaluate(test_loader, model)
        #print(score)
        current_loss = step_loss / step

        if (step % 2000 == 0 and step >= 3000):
        #if (step % 4000 == 0):
            model.eval()
            src, trg = test_loader.get_batch()
            pred, lengths = model.inference(src)
            score2 = get_bleu2(pred, trg, de_idx2word, lengths)
 
            score = evaluate(test_loader, model, de_word2idx)
            
            print(f"bleu score : {score} | compare bleu : {score2}")
            #torch.save(model.state_dict(), "./current_step.pt")

        if best_loss > current_loss:
            best_loss = current_loss
            torch.save(model.state_dict(), "./model.pt")

    del loss, y_pred, target, sr_batch, tr_batch
