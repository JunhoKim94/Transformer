import numpy as np
from tqdm import tqdm
import pickle
import re
import collections
from BPE import *
import time

def word_encoding(word, word2idx):
    #word = word.lower()
    word = word + "</w>"
    encode = []
    while(len(word) != 0):
        length = len(word)
        for i in range(length):
            temp = word[:(length - i)]
            if temp in word2idx:
                encode.append(word2idx[temp])
                word = word[(length - i):]
                break

        if length == len(word):
            break
        
    return encode

def corpus_span(path, common):
    '''
    data = batch x sentence 
    '''
    en_collect = collections.Counter()
    de_collect = collections.Counter()

    en_data = []
    de_data = []
    with open(path, 'r', encoding = "utf-8") as f:
        x = f.readlines()

        for line in tqdm(x):
            line = line[:-1]
            line = line.split("\t")

            line[0] = clean_str(line[0], True)
            line[1] = clean_str(line[1], True)

            en_data.append(line[0])
            de_data.append(line[1])
            
            en_collect.update(line[0].split(" "))
            de_collect.update(line[1].split(" "))
    
    en_selected = en_collect.most_common(70000)
    de_selected = de_collect.most_common(70000)

    print(len(en_selected), len(de_selected))

    en_word2idx, en_idx2word = bpe_corpus(en_selected, common)
    de_word2idx, de_idx2word = bpe_corpus(de_selected, common)

    data = {"en_data" : (en_word2idx, en_idx2word), "de_data" : (de_word2idx, de_idx2word)}

    with open("./corpus.pickle", 'wb') as f:
        pickle.dump(data,f)
    
    pair = {"en_data" : en_data, "de_data" : de_data}
    with open("./data/split/data.pickle", "wb") as f:
        pickle.dump(pair, f)
    
    return data, en_data, de_data

def wordtoid(data, word2idx, bpe = True):
    '''
    data = (1, sen)
    '''
    data = data.split()
    temp = [word2idx["<BOS>"]]
    for word in data:
        if bpe:
            encode = word_encoding(word, word2idx)
        else:
            if word in word2idx:
                encode = [word2idx[word]]
            else:
                encode = None

        if encode == None:
            continue
        temp += encode
        #temp.append(word2idx[word])
    temp.append(word2idx["<EOS>"])

    return temp

def padding(data, length):
    '''
    data = [batch_seq, length]
    '''
    l = [len(s) for s in data]
    if len(l) == 0:
        print(data)
        return np.zeros((1, 10), dtype= np.int32)

    batch_size = len(data)
    max_length = max(l)

    if max_length > length:
        max_length = length

    batch = np.zeros((batch_size, max_length), dtype = np.int32)

    for i in range(len(data)):
        if l[i] > length:
            l[i] = length
        batch[i, :l[i]] = data[i][:l[i]]

    return batch

def clean_str(string, TREC = False):
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
    data, en_data, de_data = corpus_span("./data/en-de_full.txt", 37000)
    #en_word2idx, en_idx2word = data["en_data"]
    #de_word2idx, de_idx2word = data["de_data"]
