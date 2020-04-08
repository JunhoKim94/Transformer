import numpy as np 
import pickle
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
from preprocess import *
import random


def call_data(path):

    with open(path , 'rb') as f:
        corpus = pickle.load(f)
        en_word2idx, en_idx2word = corpus["en_data"]
        de_word2idx, de_idx2word = corpus["de_data"]

    return en_word2idx, en_idx2word, de_word2idx, de_idx2word

def call_train_data(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def evalutate(data, model, idx2word):

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

class Batch_Maker(Dataset):
    def __init__(self, path, en_word2idx, de_word2dix, device, max_len):
        '''
        en_data, de_data ==> raw data which distinguished by sentence
        '''
        with open(path , "rb") as f:
            data = pickle.load(f)
            self.en_data, self.de_data = data["en_data"], data["de_data"]
        #self.en_data = en_data
        #self.de_data = de_data
        self.en_word2idx = en_word2idx
        self.de_word2idx = de_word2dix
        self.max_len = max_len

        self.device = device

    def __len__(self):
        return len(self.en_data)

    def getitem(self, batch):

        sr_batch = self.get_random_sample(self.en_data, batch)
        tr_batch = self.get_random_sample(self.de_data, batch)

        sr_batch = wordtoid(sr_batch, self.en_word2idx)
        tr_batch = wordtoid(tr_batch, self.de_word2idx)

        sr_batch = padding(sr_batch, self.max_len)
        tr_batch = padding(tr_batch, self.max_len)

        sr_batch = torch.LongTensor(sr_batch).to(self.device)
        tr_batch = torch.LongTensor(tr_batch).to(self.device)

        return sr_batch, tr_batch

    def get_random_sample(self, data, batch):
        return [data[random.randint(0, len(data))] for _ in range(batch)]
