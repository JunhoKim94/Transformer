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

PAD = 0
lr = 0.0025
steps = 300000
warm_up = 12000

print("\n ==============================> Training Start <=============================")
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
#device = torch.device("cpu")
print(torch.cuda.is_available())
corpus_path = "./corpus.pickle"
data_path = "./data/split/data.pickle"

#data, en_data, de_data = corpus_span(path, 50000)
en_word2idx, en_idx2word, de_word2idx, de_idx2word = call_data(corpus_path)

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
dropout = 0.2
max_len = 300
h = 8
Num = 6
max_token = 2800

#dataset
dataset = Basedataset(data_path, en_word2idx, de_word2idx)
dataloader = Batch_loader(dataset, device, max_len, max_token)

total = len(dataset)

#model
encoder = Encoder(en_vocab_size, emb_size , d_ff, dropout, max_len, h, Num, device)
decoder = Decoder(de_vocab_size, emb_size , d_ff, dropout, max_len, h, Num // 2, device)
model = Transformer(encoder, decoder, PAD, device).to(device)

#model.load_state_dict(torch.load("./current_step.pt"))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = lr, betas = (0.9, 0.98), eps = 1e-9)

#d_model ** -0.5 
#scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer = optimizer, lr_lambda = lambda epoch : 0.95**epoch)

test_dataset = Basedataset("./data/split/test.pickle", en_word2idx, de_word2idx)
test_loader = Batch_loader(test_dataset, device, max_len, max_token)


'''
model.eval()
src, trg = test_loader.get_batch()
pred = model.inference(src, 10)
score = get_bleu(pred, trg, de_idx2word)
print(score)
'''

best_loss = 1e5
st = time.time()

step_loss = 0
avg_batch = 0
avg_src_seq = 0
avg_trg_seq = 0
start = 0
for step in range(start + 1, start + steps+1):
    model.train()

    for param_group in optimizer.param_groups:
        param_group['lr'] =  emb_size**(-0.5) * min(step**(-0.5), step * (warm_up**(-1.5))) / 16
        lr = param_group['lr']

    sr_batch, tr_batch = dataloader.get_batch()

    target = tr_batch[:,1:]
    tr_batch = tr_batch[:,:-1]

    target = target.contiguous().view(-1)
    #print(b_target, target)

    y_pred = model(sr_batch, tr_batch)
    #y_pred = y_pred.transpose(1,2)
    loss = criterion(y_pred, target)

    #torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    step_loss += loss.item()
    avg_batch += len(sr_batch)
    avg_src_seq += sr_batch.size(1)
    avg_trg_seq += tr_batch.size(1)

    if step % 1000 == 0:
        d = step + 1e-5 - start
        print(f"total step : {steps + start}  |  curr_step : {step}  |  Time Spend : {(time.time() - st) / 3600} hours  | loss :  { step_loss / d} | lr : {lr} | avg_batch : {int(avg_batch / d)} | avg_src_seq : {int(avg_src_seq / d)} | avg_trg_seq : {int(avg_trg_seq / d)}")

        if step % 2000 == 0:
            model.eval()
            src, trg = test_loader.get_batch()
            pred = model.inference(src, 10)
            score = get_bleu(pred, trg, de_idx2word)
            
            print(f"bleu score : {score}")
            torch.save(model.state_dict(), "./current_step.pt")

        if best_loss > step_loss:
            best_loss = step_loss
            torch.save(model.state_dict(), "./model.pt")

    del loss, y_pred, target, sr_batch, tr_batch


        #test = torch.Tensor(get_mini(b_train, 1)).to(torch.long).to(device)
        #evaluate(test, idx2word)
        #ret = model.generate(test, 10, None, 5)
        #evaluate(ret, idx2word)
