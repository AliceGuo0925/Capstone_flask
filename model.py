# import packages
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


# modeling
model = BertModel.from_pretrained('clinical_bert', output_attentions=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = BertTokenizer.from_pretrained('clinical_bert', do_lower_case=True)
#text=input("please enter clinical note you want to analyze:")
MASK_VALUE = -1e18


class HAN_bert(nn.Module):
    def __init__(self, bert_model, C, E=768, h=50, L=200, T=50, bidirectional=True, dropout=0.5):
        super().__init__()
        """
        E embedding size of bert 768
        C number of class
        B batch size
        L max number of sentence in a documnet 200
        T max number of words in a sentence 50
        h hidden dimension of GRU
        """
        self.C = C
        self.L = L
        self.h = h
        self.H = 2 * h if bidirectional else h
        self.T = T
        self.E = E

        self.word_encoder = bert_model
        self.word_linear = nn.Linear(in_features=self.E, out_features=self.H)
        self.sentence_encoder = nn.GRU(
            input_size=self.H, hidden_size=h, bidirectional=bidirectional, batch_first=True)
        self.sentence_attention = nn.Parameter(
            torch.randn([self.H, 1]).float())
        self.sentence_linear = nn.Linear(
            in_features=self.H, out_features=self.H)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.H, self.C)

    def forward(self, paragraph_list):
        paragraph_list = paragraph_list[0]
        B = paragraph_list.shape[0]
        # print(paragraph_list)
        #B = paragraph_list.shape[0]
        words_mask = paragraph_list != 0
        words_mask = words_mask.view(B * self.L, self.T)
        #print("pass view")
        words_mask = words_mask.unsqueeze(-1)
        # paragraph_list: [batch size, num of sentence, num of word]
        paragraph_list = paragraph_list.view(B * self.L, self.T)
        # [batch size*num of sentence, num of word]
        emb_paragraph_list = self.word_encoder(paragraph_list)
        # [batch size* num of sentence, num of word, 768]
        h_it = emb_paragraph_list[0]
        # batch_size*number of sentence, sequence_length, 768
        word_attn = emb_paragraph_list[2][0]  # tuple
        # [batch_size*num of sentence, num_heads, sequence_length, sequence_length]
        u_it = torch.tanh(self.word_linear(h_it))
        # [batch size*number of sentence, num of word, H]
        x = torch.mean(u_it, dim=1).squeeze(1).view(B, self.L, self.H)
        # [batch size, num of sentence, H]

        h_i, _ = self.sentence_encoder(x)
        # [batch size, L, H]
        u_i = torch.tanh(self.sentence_linear(h_i))
        # [batch size, L, H]
        u_s = self.sentence_attention.unsqueeze(0).expand(B, -1, -1)
        # [batch size, H, 1]
        v = u_i.bmm(u_s)
        # [batch size, L, 1]
        words_mask = words_mask.view(B, self.L, -1)
        # [batch size, L, T]
        sents_mask = words_mask.sum(-1) != 0
        # [batch size, L]
        sents_mask = sents_mask.unsqueeze(-1)
        # [batch size, L, 1]

        v = v.masked_fill(~sents_mask, MASK_VALUE)
        a_i = torch.softmax(v, dim=1)
        x_ = a_i.transpose(-1, -2).bmm(h_i)
        # [batch size, 1, H]
        x = h_i.mean(dim=1).unsqueeze(dim=1) + x_
        x = x.squeeze(1)
        # [batch size, H]

        x = self.dropout(x)
        logits = self.linear(x)
        # [batch size, C]
        # logits=torch.sigmoid(logits)
        word_attn = word_attn.view(B, self.L, 12, self.T, self.T)
        h_it = h_it.view(B, self.L, self.T, -1)
        return logits, a_i.squeeze(2), word_attn, h_it


han_bert = HAN_bert(model, 6585)
han_bert.load_state_dict(torch.load(
    'params_new1.pt', map_location=torch.device('cpu')), strict=False)
han_bert.to(device)
pickle.dump(han_bert, open('model.pkl', 'wb'))
