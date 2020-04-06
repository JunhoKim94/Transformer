import numpy as np
import torch
import torch.nn as nn
import pickle
from preprocess import *
from model.model import Encoder, Decoder, Transformer
import time
import matplotlib.pyplot as plt

PAD = 0
batch = 32
lr = 0.001
epochs = 8
total = 22520376

print("\n ==============================> Training Start <=============================")
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
#device = torch.device("cpu")
print(torch.cuda.is_available())
path = ["./en_corpus.pickle", "./fr_corpus.pickle"]

def call_data(paths):
    data = []
    for path in paths:
        with open(path , 'rb') as f:
            corpus = pickle.load(f)
            word2idx = corpus["word2idx"]
            idx2word = corpus["idx2word"]
        data.append((word2idx, idx2word))

    return data

data = call_data(path)
en_word2idx, en_idx2word = data[0]
fr_word2idx, fr_idx2word = data[1]

print(len(en_word2idx) , len(fr_word2idx))

en_vocab_size = len(en_word2idx)
fr_vocab_size = len(fr_word2idx)
emb_size = 512
d_ff = 2048
dropout = 0.2
max_len = 50
h = 8
Num = 6

encoder = Encoder(en_vocab_size, emb_size , d_ff, dropout, max_len, h, Num, device)
decoder = Decoder(fr_vocab_size, emb_size , d_ff, dropout, max_len, h, Num / 2, device)
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
    
    for i in range(20):
        en_data = call_train_data("C:/Users/dilab/Documents/GitHub/Seq2Seq/data/en_data_%d.pickle"%i)
        fr_data = call_train_data("C:/Users/dilab/Documents/GitHub/Seq2Seq/data/fr_data_%d.pickle"%i)

        sub_total = len(en_data) // batch
        #print(len(en_data), len(fr_data))
        for iteration in range(sub_total):
            #seed = np.random.choice(len(en_data), batch)
            b_train = en_data[iteration * batch : (iteration + 1) * batch]
            b_target = fr_data[iteration * batch : (iteration + 1) * batch]

            b_train = wordtoid(b_train, en_word2idx)
            b_train = torch.Tensor(padding(b_train, max_len)).to(torch.long).to(device)

            b_target = wordtoid(b_target, fr_word2idx)
            b_target = torch.Tensor(padding(b_target, max_len)).to(torch.long).to(device)

            target = b_target[:,1:]
            b_target = b_target[:,:-1]

            target = target.contiguous().view(-1)
            #print(b_target, target)
            y_pred = model(b_train, b_target)
            loss = criterion(y_pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if iteration % 500 == 0:
                print(f"current file : {i}  |  total iteration : {sub_total}  |  iteration : {iteration}  |  Time Spend : {(time.time() - st) / 3600} hours  | loss :  { epoch_loss / (i * sub_total + iteration + 1e-5)}")
                torch.save(model.state_dict(), "./current_iter.pt")

            del loss, y_pred, b_train, b_target, target

    epoch_loss /= 20 * sub_total

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