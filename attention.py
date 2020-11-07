import matplotlib.pyplot as plt
from IPython.core.display import display, HTML
import seaborn as sns
import pandas as pd
from flask import Flask, request, jsonify, render_template, flash
from wtforms import Form, validators, StringField
from preprocessing import preprocess1, preprocessing, convert_text_tensor
from prediction import prepare_dataloader, evaluation, prediction_probability
from model import HAN_bert
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
import html
from sklearn.pipeline import FeatureUnion, Pipeline
warnings.filterwarnings("ignore")

model = BertModel.from_pretrained('clinical_bert', output_attentions=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = BertTokenizer.from_pretrained('clinical_bert', do_lower_case=True)
han_bert = pickle.load(open('model.pkl', 'rb'))


def rescale(input_list):
    the_array = np.asarray(input_list)
    the_max = np.max(the_array)
    the_min = np.min(the_array)
    rescale = (the_array - the_min)/(the_max-the_min)*3
    return rescale.tolist()


def rescale_1(input_list):
    the_array = np.asarray(input_list)
    the_max = np.max(the_array)
    the_min = np.min(the_array)
    rescale = (the_array - the_min)/(the_max-the_min)
    return rescale.tolist()


def html_escape(text):
    return html.escape(text)


def highlight(text, paragraph_list_id, s_attn, w_attn2):
    complete_text = []
    for i in range(200):
        sentence = tokenizer.convert_ids_to_tokens(paragraph_list_id[0][i])
        complete_text += sentence

    w_attn3 = []
    for i in range(200):
        w_attn3.append(rescale(np.sum(w_attn2[0], axis=2)[i]))

    all_attn = []
    for i in range(200):
        all_attn += [j*rescale_1(s_attn[0])[i] for j in w_attn3[i]]

    all_attn = rescale_1(all_attn)

    max_alpha = 1
    highlighted_text = []
    for i in range(len(complete_text)):
        word = complete_text[i]
        if word not in ['[SEP]', '<', '>', 'pad', '[CLS]', '[PAD]']:
            weight = all_attn[i]
            highlighted_text.append('<span style="background-color:rgba(135,206,250,' + str(
                weight * max_alpha) + ');" > ' + html_escape(word) + ' </span >')
        else:
            pass
    highlighted_text = " ".join(highlighted_text)
    return highlighted_text
