import pickle
from preprocess import *
import random
import torch
from torch.utils.data import Dataset, DataLoader

class Basedataset(Dataset):
    def __init__(self, path, en_word2idx, de_word2idx):
        '''
        en_data, de_data ==> raw data which distinguished by sentence
        '''
        with open(path , "rb") as f:
            data = pickle.load(f)
            self.en_data, self.de_data = data["en_data"], data["de_data"]
        #self.en_data = en_data
        #self.de_data = de_data
        self.en_word2idx = en_word2idx
        self.de_word2idx = de_word2idx

    def __len__(self):
        return len(self.en_data)

    def __getitem__(self, idx):

        #sr_batch = self.get_random_sample(self.en_data, batch)
        #tr_batch = self.get_random_sample(self.de_data, batch)

        sr_batch = wordtoid(self.en_data[idx], self.en_word2idx)
        tr_batch = wordtoid(self.de_data[idx], self.de_word2idx)

        return sr_batch, tr_batch

    def get_random_sample(self, data, batch):
        return [data[random.randint(0, len(data))] for _ in range(batch)]

class Batch_loader:
    def __init__(self, dataset, device, max_len, max_token):
        self.dataset = dataset
        self.device =device
        self.max_len = max_len
        self.max_token = max_token


    def __len__(self):
        return len(self.dataset)


    def get_batch(self):
        
        length = len(self.dataset)

        max_src, max_trg, sen_len = 0, 0, 0
        sr_batch, tr_batch = [],[]
        while(1):
            seed = random.randint(0, length - 1)
            src, trg = self.dataset[seed]
            
            sen_len += 1
            max_src = len(src) if max_src < len(src) else max_src
            max_trg = len(trg) if max_trg < len(trg) else max_trg

            if (max_src * sen_len > self.max_token) | (max_trg * sen_len > self.max_token):
                break
            sr_batch.append(src)
            tr_batch.append(trg)

        if max_src > self.max_len:
            max_src = self.max_len
        
        if max_trg > self.max_len:
            max_trg = self.max_len

        sr_batch = padding(sr_batch, max_src, sen_len)
        tr_batch = padding(tr_batch, max_trg, sen_len)
        #print(sr_batch.shape, tr_batch.shape, sen_len)
        sr_batch = torch.LongTensor(sr_batch).to(self.device)
        tr_batch = torch.LongTensor(tr_batch).to(self.device)

        return sr_batch, tr_batch