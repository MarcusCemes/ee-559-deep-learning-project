# -*- coding: utf-8 -*-
"""EE559Project.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1g6oeCU3J-M7qIySEtroetlDrh04IvqOc
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
import re
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
from collections import defaultdict
import time
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split



## get the model from the huggingface model hub
#!git clone https://huggingface.co/GroNLP/hateBERT
## If you download the model locally , you can load it from the local path
PATH = os.getcwd()+"/hateBERT"

## Load the BERT model
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained(PATH)
model = BertModel.from_pretrained(PATH)

print(model)

# Load the model weights using hugingface model hub
from transformers import BertTokenizer, BertModel
import torch

# Example of getting the output of the model for a given text
def tokenize_text(text):
    return tokenizer(text, padding=True, truncation=True, return_tensors="pt")

# Use the model in inference mode and classify a give example
def classify(text):
    inputs = tokenize_text(text)
    print(inputs)
    outputs = model(**inputs)
    return outputs

text = "Hello World"
outputs = classify(text)

print(outputs)

# Download the data locally
# https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech/tree/main

## Read the measuring-hate-speech.parquet
parquet_data = pd.read_parquet('measuring-hate-speech.parquet')

## remove from df the redundant  columns
multilabel_dataset = parquet_data.iloc[:, :-116]
# multilabel_dataset = parquet_data.iloc[:15, :-116]

## Remove comment_idm,annotator_id,platform and put the column text in the first position
multilabel_dataset = multilabel_dataset[['text', 'hatespeech', 'sentiment', 'hate_speech_score', 'respect','insult','humiliate','status','dehumanize','violence','genocide' ,'attack_defend']]
multilabel_dataset = multilabel_dataset.iloc[:, 0:2]
# Apply the transformation
multilabel_dataset['hatespeech'] = multilabel_dataset['hatespeech'].apply(lambda x: 1 if x > 0 else 0)
multilabel_dataset.rename(columns={'hatespeech': 'labels'}, inplace=True)


#Read the new dataset
new_df = pd.read_excel('yeni_data.xlsx')
new_df = new_df[['message', 'Class']]
new_df.rename(columns={'message': 'text', 'Class': 'labels'}, inplace=True)
new_df['labels'] = new_df['labels'].apply(lambda x: 1 if x == 'Hateful' else 0)
new_df

concatenated_df = pd.concat([new_df, multilabel_dataset], ignore_index=True)
concatenated_df = concatenated_df[concatenated_df['text'] != '']
concatenated_df = concatenated_df[concatenated_df['text'].apply(lambda x: isinstance(x, str))]
concatenated_df['labels']=concatenated_df['labels'].values.tolist()
# Function to convert label to list
def convert_label(label):
    if label == 1:
        return [0, 1] ##means hate speech
    elif label == 0:
        return [1, 0]  ##means no hate speech
    else:
        raise ValueError("Label must be either 0 or 1")

# Apply the function to the label column and create a new column
concatenated_df['labels'] = concatenated_df['labels'].apply(convert_label)


## Preprocess the TEXT data
## Remove HTML tags
def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

concatenated_df['text'] = concatenated_df['text'].apply(lambda x: remove_html_tags(x))

## Remove URL
def remove_url(text):
    return re.sub(r'http\S+', '', text)

concatenated_df['text'] = concatenated_df['text'].apply(lambda x: remove_url(x))

## Lowercase
concatenated_df['text'] = concatenated_df['text'].str.lower()

## Remove punctuation
def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

concatenated_df['text'] = concatenated_df['text'].apply(lambda x: remove_punctuation(x))

## Handling Contractions using libraries

def expand_contractions(text):
    return contractions.fix(text)

concatenated_df['text'] = concatenated_df['text'].apply(lambda x: expand_contractions(x))

## Remove stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in stop_words])

concatenated_df['text'] = concatenated_df['text'].apply(lambda x: remove_stopwords(x))

## Lemmatization
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def lemmatize_words(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])
concatenated_df['text'] = concatenated_df['text'].apply(lambda x: lemmatize_words(x))





MAX_LEN = 256
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 8
EPOCHS = 1
LEARNING_RATE = 1e-05

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

import torch
from torch.utils.data import Dataset

class HateSpeechDataset(Dataset):
    def __init__(self, data, tokenizer,max_len=256):

        self.tokenizer = tokenizer
        self.data = data
        self.text = data.text
        self.targets = self.data.labels
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        text = str(self.text.iloc[idx])

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True
        )


        ids = encoding['input_ids']
        mask = encoding['attention_mask']
        token_type_ids = encoding['token_type_ids']

        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets.iloc[idx], dtype=torch.float).to(device)
        }

train_texts, val_texts,= train_test_split(concatenated_df,test_size=0.2)

print("Dataset length: {}".format(concatenated_df.shape))
print("Train Dataset length: {}".format(train_texts.shape))
print("Val Dataset length: {}".format(val_texts.shape))

training_dataset = HateSpeechDataset(train_texts,tokenizer,MAX_LEN)
validation_dataset = HateSpeechDataset(val_texts,tokenizer,MAX_LEN)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_dataset, **train_params)
validation_loader = DataLoader(validation_dataset, **test_params)

next(iter(training_loader))

print(model)

# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output for the model.

class MultilabelHateBert(torch.nn.Module):
    def __init__(self,bertmodel):
        super(MultilabelHateBert, self).__init__()
        self.bertmodel = bertmodel
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, 2)

    def forward(self, ids, mask, token_type_ids):

      output_1= self.bertmodel(ids, attention_mask = mask, token_type_ids = token_type_ids)
      output_2 = self.dropout(output_1.pooler_output)
      output = self.linear(output_2)
      return output

multilabel_model = MultilabelHateBert(model)
multilabel_model.to(device)

def criterion(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

optimizer = torch.optim.Adam(params =  multilabel_model.parameters(), lr=LEARNING_RATE)

results_models_weights_dir = 'models_weights_with_bert/'
if not os.path.exists(results_models_weights_dir):
    os.mkdir(results_models_weights_dir)

### Osman Changed this part
from tqdm import tqdm
from sklearn import metrics

total_loss = 0.0
batch_count = 0

for epoch in tqdm(range(EPOCHS), desc="Epochs"):

  multilabel_model.train()

  for i, batch in tqdm(enumerate(training_loader), desc=f"Epoch {epoch + 1}", total=len(training_loader)):

    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    token_type_ids = batch['token_type_ids'].to(device)
    targets = batch['targets'].to(device)
    #I moved optimizer.zero_grad() to here, and deleted one of them
    optimizer.zero_grad()
    outputs = multilabel_model(input_ids,attention_mask,token_type_ids)
    #optimizer.zero_grad()
    print("Outputs:",outputs)
    print("Targets:",targets)
    loss = criterion(outputs,targets)

    total_loss += loss.item()
    batch_count += 1
    if i%50==0:
      print()
      print(f'Epoch: {epoch}, Loss{total_loss/batch_count}')

    #optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  torch.save(multilabel_model.state_dict(), results_models_weights_dir + 'base_model_binary_bert.pth')
  print("Model state saved for Epoch:", epoch)


  ####Evaluationnn
  multilabel_model.eval()
  val_targets=[]
  val_outputs=[]

  with torch.no_grad():

    for i,batch in enumerate(validation_loader):

      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)
      token_type_ids = batch['token_type_ids'].to(device)
      targets = batch['targets'].to(device)

      outputs = multilabel_model(input_ids,attention_mask,token_type_ids)

      val_targets.extend(targets.cpu().detach().numpy().tolist())
      val_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

  outputs = np.array(val_outputs) >= 0.5
  accuracy = metrics.accuracy_score(val_targets, outputs)
  f1_score_micro = metrics.f1_score(val_targets, outputs, average='micro')
  f1_score_macro = metrics.f1_score(val_targets, outputs, average='macro')
  print(f"Accuracy Score = {accuracy}")
  print(f"F1 Score (Micro) = {f1_score_micro}")
  print(f"F1 Score (Macro) = {f1_score_macro}")