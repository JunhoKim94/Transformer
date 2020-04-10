import numpy as np 
import pickle
import matplotlib.pyplot as plt
import torch
from preprocess import *
import random
import re, collections

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