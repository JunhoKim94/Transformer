import numpy as np
from tqdm import tqdm
import pickle
import re
import collections

def word_encoding(word, word2idx):
    word = word + "</w>"

    encode = []
    while(len(word) != 0):
        length = len(word)
        for i in range(length - 1):
            temp = word[:(length - i)]
            if temp in word2idx:
                encode.append(word2idx[temp])
                word = word[(length - i):]
                break
        '''
        if len(encode) == 0:
            encode.append(word2idx["PAD"])
            break
        '''
    return encode
    

def corpus_span(path, common):
    '''
    data = batch x sentence 
    '''
    en_word2idx = {"PAD" : 0 ,"<BOS>" : 1, "<EOS>" : 2}
    en_idx2word = {0 : "PAD", 1 : "<BOS>", 2: "<EOS>"}

    de_word2idx = {"PAD" : 0 ,"<BOS>" : 1, "<EOS>" : 2}
    de_idx2word = {0 : "PAD", 1 : "<BOS>", 2: "<EOS>"}

    en_collect = collections.Counter()
    de_collect = collections.Counter()

    en_data = []
    de_data = []
    with open(path, 'r', encoding = "utf-8") as f:
        x = f.readlines()

        for line in tqdm(x):
            line = line[:-1]
            line = line.split("\t")

            en_data.append(line[0])
            de_data.append(line[1])
            
            en_collect.update(line[0].split(" "))
            de_collect.update(line[1].split(" "))

    en_selected = en_collect.most_common(common)
    de_selected = de_collect.most_common(common)

    for word, freq in en_selected:
        en_word2idx[word] = len(en_word2idx)
        en_idx2word[len(en_idx2word)] = word



    for word, freq in de_selected:
        de_word2idx[word] = len(de_word2idx)
        de_idx2word[len(de_idx2word)] = word

    data = {"en_data" : (en_word2idx, en_idx2word), "de_data" : (de_word2idx, de_idx2word)}

    with open("./corpus.pickle", 'wb') as f:
        pickle.dump(data,f)

    pair = {"en_data" : en_data, "de_data" : de_data}
    with open("./data/split/data.pickle", "wb") as f:
        pickle.dump(pair, f)

    return data, en_data, de_data

def wordtoid(data, word2idx):
    '''
    data = (B, sen)
    '''
    train_data = []

    for line in data:
        line = line.split()
        temp = [word2idx["<BOS>"]]
        for word in line:

            encode = word_encoding(word, word2idx)
            #if word not in word2idx:
                #temp.append(word2idx["<UNK>"])
                #continue
            temp += encode
            #temp.append(word2idx[word])
        temp.append(word2idx["<EOS>"])
        train_data.append(temp)

    return train_data

def padding(data, length):
    '''
    data = [batch_seq, length]
    '''
    l = [len(s) for s in data]
    max_length = max(l)

    if max_length > length:
        max_length = length

    batch = np.zeros((len(data), max_length), dtype = np.int32)

    for i in range(len(data)):
        if l[i] > length:
            l[i] = length
        batch[i, :l[i]] = data[i][:l[i]]

    return batch

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
    data, en_data, de_data = corpus_span("./data/en-de_full.txt", 50000)
    print(en_data)
    print(de_data)