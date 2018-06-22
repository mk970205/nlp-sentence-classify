import re
import numpy as np
def load_data(pos_data_dir, neg_data_dir):
    #file read
    pos_text = open(pos_data_dir, "r").readlines()
    neg_text = open(neg_data_dir, "r").readlines()

    # pos -> label = [0, 1] 
    # neg -> label = [1, 0]
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
