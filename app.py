import matplotlib.pyplot as plt
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
from sklearn.pipeline import FeatureUnion, Pipeline
warnings.filterwarnings("ignore")

app = Flask('icd10_prediction')
model = BertModel.from_pretrained('clinical_bert', output_attentions=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = BertTokenizer.from_pretrained('clinical_bert', do_lower_case=True)
han_bert = pickle.load(open('model.pkl', 'rb'))
df_all = pd.read_csv("icd10_lookup.csv", index_col='icd10_code')


class InputForm(Form):
    clinical_text = StringField(validators=[validators.required()])


num1 = 0
df_html = 0
finding = 0


@app.route('/', methods=['POST', 'GET'])
def predict():
    global num1
    global df_html
    global finding
    #form = InputForm(request.form)
    # form=request.form['clinical_text']
    if request.method == 'POST':
        finding = request.form['clinical_text']

        # if form.validate():
        if len(finding) > 0:
            flash('Generating model predictions...')
            text = preprocess1(finding)
            text = np.array([text])
            text = pd.Series(text)
            text = preprocessing(text)
            text = text[0]
            paragraph_list_id, paragraph_mask_id = convert_text_tensor(text)
            dataloader = prepare_dataloader(paragraph_list_id)
            #print([i for i in dataloader])
            s_attn, pred, w_attn, w_attn2 = evaluation(dataloader, han_bert)
            num1, df = prediction_probability(s_attn, pred, w_attn, w_attn2)
            df.loc[:, 'ICD-10 code'].replace(".", "")
            df = df.set_index("ICD-10 code")
            df_final = pd.concat([df_all, df], axis=1, join='inner')
            df_final = df_final.sort_values(
                by=['probability score'], ascending=False)
            df_final = df_final.rename(
                columns={"combine": "ICD-10 description", "probability score": "probability"})
            df_final = df_final.reset_index()
            df_final = df_final.rename(columns={"index": "ICD-10 code"})
            df_html = df_final.to_html(index=False, col_space=150). \
                replace('<tr>', '<tr style="text-align: center;">'). \
                replace('<th style="', '<th style="text-align: center;')

        else:
            flash('Error: the form of clinical text is not correct.')
        #global num1

    return render_template('index.html', form=finding, number=num1, tables=[df_html])


if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(debug=True, host='0.0.0.0', port=5550)
