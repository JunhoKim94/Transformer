import numpy as np
import torch
import torch.nn as nn
import pickle
from preprocess import *
from model.model import Encoder, Decoder, Transformer
import time
import matplotlib.pyplot as plt
from utils import *

PAD = 0
batch = 32
lr = 0.001
epochs = 8

print("\n ==============================> Training Start <=============================")
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
#device = torch.device("cpu")
print(torch.cuda.is_available())
corpus_path = "./corpus.pickle"
data_path = "./data/split/data.pickle"

#data, en_data, de_data = corpus_span(path, 50000)
en_word2idx, en_idx2word, de_word2idx, de_idx2word = call_data(corpus_path)

print(en_word2idx, de_word2idx)

print(len(en_word2idx) , len(de_word2idx))

en_vocab_size = len(en_word2idx)
de_vocab_size = len(de_word2idx)
emb_size = 512
d_ff = 2048
dropout = 0.2
max_len = 50
h = 8
Num = 6

#dataset
dataset = Batch_Maker(data_path, en_word2idx, de_word2idx, device, max_len)

total = len(dataset)

#model
encoder = Encoder(en_vocab_size, emb_size , d_ff, dropout, max_len, h, Num, device)
decoder = Decoder(de_vocab_size, emb_size , d_ff, dropout, max_len, h, Num / 2, device)
model = Transformer(encoder, decoder, PAD, device)

#model.load_state_dict(torch.load("./model.pt"))
criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr = lr)
optimizer = torch.optim.Adam(model.parameters(), lr = lr, betas = (0.9, 0.98), eps = 1e-9)
torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

#d_model ** -0.5 
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.5**epoch)

model.train()
model.to(device)

def call_train_data(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

best_loss = 1e5
st = time.time()

for epoch in range(epochs):
    epoch_loss = 0
    for iteration in range(total// batch):
        #seed = np.random.choice(len(en_data), batch)
        sr_batch, tr_batch = dataset.getitem(batch)

        target = tr_batch[:,1:]
        #b_target = b_target[:,:-1]
        target = target.contiguous().view(-1)
        #print(b_target, target)
        y_pred = model(sr_batch, tr_batch)
        loss = criterion(y_pred, target)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()


        if iteration % 500 == 0:
            print(f"total iteration : {total // batch}  |  iteration : {iteration}  |  Time Spend : {(time.time() - st) / 3600} hours  | loss :  { epoch_loss / (iteration + 1e-5)}")
            torch.save(model.state_dict(), "./current_iter.pt")

        del loss, y_pred, target

    epoch_loss /= (total // batch)

    if epoch >= 4:
        scheduler.step()
    if epoch % 1 == 0:
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        #test = torch.Tensor(get_mini(b_train, 1)).to(torch.long).to(device)
        #evaluate(test, idx2word)
        #ret = model.generate(test, 10, None, 5)
        #evaluate(ret, idx2word)
        print(f"epoch : {epoch}  |  Epoch loss : {epoch_loss} | Time Spended : {(time.time() - st) / 3600} hours  | learning_rate : {lr}")
        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), "./model.pt")


'''
en_test = call_train_data("./data/en_data_3.pickle")
en_test = en_test[20 : 40]
en_test = wordtoid(en_test, en_word2idx)
en_test = torch.Tensor(padding(en_test, max_len)).to(torch.long).to(device)

def evalutate(data, model, idx2word):

    model.load_state_dict(torch.load("./model.pt"))
    model.eval()
    ret, attn = model.generate(data, 15, None)
    attn = np.array(attn.to("cpu").detach())
    plt.imshow(attn[0])
    plt.show()
    gen = []
    for line in ret:
        temp = []
        for word in line:
            if word.item() not in idx2word:
                continue
            temp += [idx2word[word.item()]]
        gen.append(temp)
    return gen

gen = evalutate(en_test, model, fr_idx2word)
print(gen)
'''