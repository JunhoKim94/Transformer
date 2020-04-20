import numpy as np 
import pickle
import matplotlib.pyplot as plt
import torch
from preprocess import *
import random
import re, collections
import nltk.translate.bleu_score as bleu

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

def evalutate(dataloader, model):
    model.eval()
    total_loss = 0.
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        sr_batch , tr_batch = dataloader.get_batch()

        target = tr_batch[:,1:]
        tr_batch = tr_batch[:,:-1]

        target = target.contiguous().view(-1)
        #print(b_target, target)
        y_pred = model(sr_batch, tr_batch)
        loss = criterion(y_pred, target)
        loss = loss * len(sr_batch)

    return total_loss / len(dataloader)

def get_bleu(pred, trg, trg_idx2word):
    '''
    pred = (B, S)
    trg = (B, S)
    '''
    pred = pred.cpu().numpy()
    trg = trg.cpu().numpy()
    batch = pred.shape[0]

    cc = bleu.SmoothingFunction()
    
    score = 0
    for p,t in zip(pred, trg):
        s1 = []
        s2 = []

        p = p[p != 0]
        t = t[t != 0]
        for idx in p:
            if idx in trg_idx2word:
                s1.append(trg_idx2word[idx])
        for idx in t:
            if idx in trg_idx2word:
                s2.append(trg_idx2word[idx])
        print("predict : ", s1)
        print("target : ", s2)
        if len(t) == 0:
            batch -= 1
            continue
        score += bleu.sentence_bleu([t],p, [1,0,0,0], smoothing_function= cc.method1)

    return score / batch