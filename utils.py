import numpy as np 
import pickle
import matplotlib.pyplot as plt
import torch
from preprocess import *
import random
import re, collections
import nltk.translate.bleu_score as bleu

def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
        loss /= pred.shape[0]
    else:
        loss = F.cross_entropy(pred, gold, ignore_index= trg_pad_idx, reduction='sum')
    return loss



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


def get_bleu2(pred, trg, trg_idx2word, lengths):
    pred = pred.cpu().numpy()
    trg = trg.cpu().numpy()
    batch = pred.shape[0]

    cc = bleu.SmoothingFunction()
    
    score = 0
    b = 0
    for p,t in zip(pred, trg):
        p = p[1:lengths[b]]
        t = t[t != 0]
        t = t[1:-1]
        b += 1
        print("predict : ", p)
        print("target : ", t)
        if len(t) == 0:
            batch -= 1
            continue
        score += bleu.sentence_bleu([t],p, [0.25,0.25,0.25,0.25], smoothing_function= cc.method1)

    return score / batch * 100


def get_bleu(pred, trg, lengths, idx2word, BLEU = [0.25, 0.25, 0.25, 0.25]):

    batch = pred.shape[0]

    cc = bleu.SmoothingFunction()
    
    score = 0
    b = 0
    for p,t in zip(pred, trg):
        p = p[1:lengths[b]]
        p = [idx2word[index].lower() for index in p]
        t = t[t != 0]
        t = t[1:-1]
        t = [idx2word[index].lower() for index in t]
        b += 1

        #print("pred :" , p)
        #print("target: ", t)

        score += bleu.sentence_bleu([t],p, BLEU, smoothing_function= cc.method1)

    return score

def evaluate(dataloader, model, word2idx):
    
    total = 0
    bleu = 0

    idx2word = dict()
    for word in word2idx.keys():
        idx2word[len(idx2word)] = word

    while(1):
    #for idx in range(len(dataloader) // batch):
        src, trg = dataloader.get_batch(ran = False)
        total += src.shape[0]
        pred, length = model.inference(src)
        
        pred = pred.cpu().numpy()
        trg = trg.cpu().numpy()

        score = get_bleu(pred, trg, length, idx2word, BLEU = [0.25, 0.25, 0.25, 0.25])
        bleu += score

        if dataloader.idx >= len(dataloader):
            dataloader.idx = 0
            break


    bleu /= (total/100)
    bleu = round(bleu, 2)

    return bleu