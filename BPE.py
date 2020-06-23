import numpy as np
import torch
import torch.nn as nn
import pickle
from preprocess import *
from utils import *
import collections

def make_vocab(corpus):
    vocab = dict()
    tokens = {"<PAD>" : 0, "<BOS>" : 1, "<EOS>" : 2}

    for word, freq in corpus:
        temp = ""
        #word = word.lower()
        for c in word:
            temp += c + " "
            if c not in tokens:
                tokens[c] = len(tokens)
        temp += "</w>"

        vocab[temp] = freq

    return vocab, tokens


def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs


def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]

    return v_out

def bpe_corpus(corpus, iteration):
    vocabs, tokens = make_vocab(corpus)

    for _ in tqdm(range(iteration), desc = "byte-pair-encoding"):
        pairs = get_stats(vocabs)
        best = max(pairs, key = pairs.get)
        tokens["".join(best)] = len(tokens)
        #print(tokens)
        vocabs = merge_vocab(best, vocabs)
        print(len(tokens), best, pairs[best], "".join(best))

    idx2word = dict()
    for word, idx in tokens.items():
        idx2word[idx] = word

    return tokens, idx2word