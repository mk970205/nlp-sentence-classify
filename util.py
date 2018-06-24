import re
import numpy as np
import codecs
def load_data(pos_data_dir, neg_data_dir):
    pos_obj = open(pos_data_dir, "rt", encoding="UTF8")
    neg_obj = open(neg_data_dir, "rt", encoding="UTF8")
    pos_text = list(pos_obj.readlines())
    neg_text = list(neg_obj.readlines())

    pos_label = []
    neg_label = []

    for i in range(len(pos_text)):
        pos_text[i] = pos_text[i].strip()
        pos_label.append([0, 1])

    for i in range(len(neg_text)):
        neg_text[i] = neg_text[i].strip()
        neg_label.append([1, 0])

    label = np.concatenate([pos_label, neg_label], 0)

    text = pos_text + neg_text

    for i in range(len(text)):
        text[i] = clean_str(text[i])

    return text, label

def clean_str(string):
    """
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
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
    return string.strip().lower()

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

# 문서의 모든 단어가 들어 있는 리스트를 만듬(중복 제외)
def makeDictionary(filePath):
    dic = []
    
    f = open(filePath, 'r')
    while True:
        line = f.readline()
        if not line: break
        for word in line.split():
            if word not in dic:
                dic.append(word)
    f.close()
    return dic

# 문서에서 가장 단어의 개수가 많은 문장을 찾은 다음 그 문장의 단어 수를 반환
def findMaxLineLength(filePath):
    f = open(filePath, 'r')
    lineLen = 0
    maxLineLen = 0
    while True:
        line = f.readline()
        if not line: break
        lineLen=len(line.split())
        maxLineLen=max(lineLen,maxLineLen)
    return maxLineLen
