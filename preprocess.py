import numpy as np
import pandas as pd
from torch import nn,tensor
from sklearn.preprocessing import StandardScaler,RobustScaler,MinMaxScaler,MaxAbsScaler
import konlpy
from konlpy.tag import *
from tqdm import tqdm
import torch
import pickle
import re
import collections

def corpus_span(path):
    '''
    data = batch x sentence 
    '''
    word2idx = {"PAD" : 0 ,"<BOS>" : 1, "<EOS>" : 2}
    idx2word = {0 : "PAD", 1 : "<BOS>", 2: "<EOS>"}

    collect = collections.Counter()
    data = []
    with open(path, 'r', encoding = "utf-8") as f:
        x = f.readlines()
        total = len(x)
        i = 0
        j = 0
        for line in tqdm(x):
            line = line[:-1]
            data.append(line)
            line = line.split(" ")
            collect.update(line)

            i += 1
            if i % (total // 20) == 0:
                with open("./data/fr_data_%d.pickle"%j, "wb") as f:
                    pickle.dump(data, f)
                j += 1
                data = []


    selected = collect.most_common(80000)

    for word, freq in selected:
        word2idx[word] = len(word2idx)
        idx2word[len(idx2word)] = word


    test_data = {"word2idx" : word2idx, "idx2word" : idx2word}

    with open("./fr_corpus.pickle", 'wb') as f:
        pickle.dump(test_data,f)

    return word2idx, data, idx2word

def wordtoid(data, word2idx):
    '''
    data = (B, sen)
    '''
    train_data = []

    for line in data:
        line = line.split()
        temp = [word2idx["<BOS>"]]
        for word in line:
            if word not in word2idx:
                continue
            temp.append(word2idx[word])
        temp.append(word2idx["<EOS>"])
        train_data.append(temp)
    return train_data

def padding(data, max_len):
    l = [len(s) for s in data]
    max_length = max(l)

    if max_length > max_len:
        max_length = max_len

    #max_length = max_len

    length = []
    for s in l:
        if s <= max_length:
            length += [s]
        else:
            length += [max_length]

    new = np.zeros((len(data), max_length), dtype = np.int32)

    for i in range(len(data)):
        new[i, :length[i]] = data[i][:length[i]]
        #new[i, -1] = length[i]

    return new

def get_mini(data, batch):
    seed = np.random.choice(len(data), batch)
    max_length = max(data[seed, -1])

    return data[seed,:max_length]

def evaluate(ret, idx2word):
    gen = []
    for line in ret:
        temp = []
        for word in line:
            if word.item() not in idx2word:
                continue
            temp.append(idx2word[word.item()])
        gen.append(temp)
    print(gen)
    return gen


def clean_str(string, TREC= False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if TREC else string.strip().lower()

if __name__ == "__main__":
    corpus_span("C:/Users/dilab/giga-fren.release2.fixed.fr")