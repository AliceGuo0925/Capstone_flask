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

model = BertModel.from_pretrained('clinical_bert', output_attentions=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = BertTokenizer.from_pretrained('clinical_bert', do_lower_case=True)


def prepare_dataloader(paragraph_list_id):
    batch_size = 1
    dataset = TensorDataset(paragraph_list_id)
    dataloader = DataLoader(
        dataset,  # The training samples.
        sampler=RandomSampler(dataset),  # Select batches randomly
        batch_size=batch_size  # Trains with this batch size.
    )
    return dataloader

# dataloader=prepare_dataloader(paragraph_list_id)


def evaluation(dataloader, model):
    model.eval()
    pred = []
    true = []
    s_attn_list = []
    w_attn_list = []
    w_attn_list2 = []
    for x in dataloader:
        #x = x.to(device)
        # y=y.to(device)
        with torch.no_grad():
            outputs, s_attn, word_attn, word_attn2 = model(x)
        # true+=y.cpu().numpy().tolist()
        # pass
        outputs = outputs[0]
        #print("outputs", outputs)
        pred += outputs.cpu().numpy().tolist()
        #print("pred", pred)
        s_attn_list += s_attn.cpu().numpy().tolist()
        # pass
        # print(word_attn[0][0][0][0]) #12,600,12,20,20
        w_attn_list += word_attn.cpu().numpy().tolist()
        # pass
        w_attn_list2 += word_attn2.cpu().numpy().tolist()
        # pass
    # true=np.array(true)
    pred = np.array(pred)

    s_attn = np.array(s_attn_list)

    w_attn = np.array(w_attn_list)

    w_attn2 = np.array(w_attn_list2)

    return s_attn, pred, w_attn, w_attn2

#s_attn, pred, w_attn, w_attn2 = evaluation(dataloader, han_bert)


def prediction_probability(s_attn, pred, w_attn, w_attn2):
    ymax, ymin = pred.max(), pred.min()
    y_pred_norm = (pred - ymin)/(ymax - ymin)
    threshold = 0.7
    y_pred_binary = y_pred_norm.copy()
    y_pred_binary[y_pred_binary >= threshold] = 1
    y_pred_binary[y_pred_binary < threshold] = 0
    # print(y_pred_binary)
    num = sum(y_pred_binary)
    # print(num)
    icd_6585 = []
    with open("6585.txt", "r") as f:
        for line in f:
            icd_6585.append(str(line.strip()))

    d = dict(zip(icd_6585, y_pred_norm))
    #sorted_d = dict(sorted(d.items(), key=operator.itemgetter(1),reverse=True))
    a1_sorted_keys = sorted(d, key=d.get, reverse=True)
    sorted_icd_name = []
    sorted_icd_score = []
    for r in a1_sorted_keys:
        sorted_icd_name.append(r)
        sorted_icd_score.append(d[r])
    d = {'ICD-10 code': sorted_icd_name, "probability score": sorted_icd_score}
    df = pd.DataFrame(d)
    return num, df
