import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import transformers
from transformers import BertModel
from transformers import BertTokenizer
import spacy
import re
import string
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import DataLoader, TensorDataset, random_split
import pickle
import warnings
from sklearn.pipeline import FeatureUnion, Pipeline
warnings.filterwarnings("ignore")

# text preprocessing


def preprocess1(x):
    y = re.sub('\\[(.*?)\\]', '', x)  # remove de-identified brackets
    # remove 1.2. since the segmenter segments based on this
    y = re.sub('[0-9]+\.', '', y)
    y = re.sub('dr\.', 'doctor', y)
    y = re.sub('m\.d\.', 'md', y)
    y = re.sub('--|__|==', '', y)

    # remove punctuation, digits, spaces
    # y = y.translate(str.maketrans("", "", string.punctuation))
    y = y.translate(str.maketrans("", "", string.digits))
    y = " ".join(y.split())
    return y

# text=preprocess1(text)


def preprocessing(x):
    x = x.fillna(' ')
    x = x.str.lower()
    x = x.str.replace('name:', ' ')
    x = x.str.replace('\n', ' ')
    x = x.str.replace('\r', ' ')
    x = x.str.replace('h/o', 'history of')
    x = x.str.replace('w/', 'with')
    x = x.str.replace('s/p', 'status post')
    x = x.str.replace('a.m. ', '')
    x = x.str.replace('p.m. ', '')
    x = x.str.replace('w/o', 'without')
    x = x.str.replace(':', ' ')
    x = x.str.replace('#', ' ')
    x = x.str.replace('*', ' ')
    x = x.str.replace('/', ' ')
    x = x.str.replace('unit no', ' ')
    x = x.str.replace('admission date', ' ')
    x = x.str.replace('discharge date', ' ')
    x = x.str.replace('date of birth', ' ')
    x = x.str.replace('-', ' ')
    x = x.str.replace(',', '')
    x = x.str.replace('mrs.', 'mrs')
    x = x.str.replace('=', ' ')
    x = x.str.replace('+', ' ')
    x = x.str.replace('%', ' ')
    x = x.str.replace('p.o.', 'per observation')
    x = x.str.replace('b.i.d.', 'twice a day')
    x = x.str.replace('t.i.d.', 'three times a day')
    x = x.str.replace('iv.', 'intravenous')
    x = x.str.replace('schatzki\'s', 'schatzkis')
    x = x.str.replace('"', '')
    x = x.str.replace('p.r.n.', 'as needed')
    x = x.str.replace('q.d.', 'once a day')
    x = x.str.replace('q.', 'every')
    x = x.str.replace('q.', 'every')
    x = x.str.replace('everyd.', 'everyday')
    x = x.str.replace('first name', ' ')
    x = x.str.replace('last name', ' ')
    x = x.str.replace('[', ' ')
    x = x.str.replace(']', ' ')
    x = x.str.replace(')', ' ')
    x = x.str.replace('(', ' ')
    x = x.str.replace(' . ', '')
    x = x.apply(str.strip)
    x = x.str.replace('?', ' 1')
    return x

#text = preprocessing(text)


tokenizer = BertTokenizer.from_pretrained('clinical_bert', do_lower_case=True)


def convert_text_tensor(text):
    paragraph_list = []
    paragraph_mask = []
    sents = []
    list_text = text.split(".")
    if len(list_text) < 200:
        list_text = list_text + (200-len(list_text))*["<pad>"]
    else:
        list_text = list_text[:200]
    sentence_list = []
    attention_masks = []
    small_sent = []
    for sent in list_text:
        word_ids = tokenizer.encode_plus(sent, truncation=True, add_special_tokens=True,
                                         max_length=50, pad_to_max_length=True, return_tensors="pt", return_attention_mask=True)
        sentence_list.append(word_ids["input_ids"])
        attention_masks.append(word_ids["attention_mask"])
        small_sent.append(sent)
    sentence_list = torch.stack(sentence_list)
    attention_masks = torch.stack(attention_masks)
    paragraph_list.append(sentence_list)
    paragraph_mask.append(attention_masks)
    sents.append(small_sent)
    paragraph_list_id = torch.stack(paragraph_list).squeeze(2)
    paragraph_mask_id = torch.stack(paragraph_mask).squeeze(2)
    return paragraph_list_id, paragraph_mask_id

# paragraph_list_id,paragraph_mask_id=convert_text_tensor(text)
