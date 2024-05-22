
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
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
from collections import defaultdict
import time
from torch.utils.data import DataLoader, TensorDataset



# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# model_name = "piubamas/beto-contextualized-hate-speech"
# # Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSequenceClassification.from_pretrained(model_name)


PATH = os.getcwd()+"/hateBERT"

## Load the BERT model
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained(PATH)
model = BertModel.from_pretrained(PATH)

#print(model)

# Example of getting the output of the model for a given text
def tokenize_text(text):
    return tokenizer(text, padding=True, truncation=True, return_tensors="pt")

# Use the model in inference mode and classify a give example
def classify(text):
    inputs = tokenize_text(text)
    print(inputs)
    outputs = model(**inputs)
    return outputs


## Preprocess the TEXT data
## Remove HTML tags
def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


## Remove URL
def remove_url(text):
    return re.sub(r'http\S+', '', text)


## Lowercase

## Remove punctuation
def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)


## Handling Contractions using libraries

def expand_contractions(text):
    return contractions.fix(text)



## Remove stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in stop_words])


## Lemmatization
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def lemmatize_words(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])


def preprocess_text(text):
    text = remove_html_tags(text)
    text = remove_url(text)
    text = text.lower()
    text = remove_punctuation(text)
    text = expand_contractions(text)
    text = remove_stopwords(text)
    text = lemmatize_words(text)
    return text


###Here Starts the Preprocessing




# class MultilabelHateBert(torch.nn.Module):
#     def __init__(self,bertmodel):
#         super(MultilabelHateBert, self).__init__()
#         self.bertmodel = bertmodel
#         self.dropout = torch.nn.Dropout(0.3)
#         self.linear = torch.nn.Linear(9, 2)

#     def forward(self, ids):

#       output_1= self.bertmodel(ids)
#       output_2 = self.dropout(output_1.logits)
#       output = self.linear(output_2)
#       return output

class MultilabelHateBert(torch.nn.Module):
    def __init__(self,bertmodel):
        super(MultilabelHateBert, self).__init__()
        self.bertmodel = bertmodel
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, 2)

    def forward(self, ids):

      output_1= self.bertmodel(ids)
      output_2 = self.dropout(output_1.pooler_output)
      output = self.linear(output_2)
      return output

multilabel_model = MultilabelHateBert(model)
multilabel_model.load_state_dict(torch.load('models_weights_with_bert/base_model_binary_bert.pth', map_location=torch.device('cpu')))

columns_to_transform = ['not_hate', 'hate']

one = "omg ego young young inexperienced woman display remarkable self absorbed proud mistake constituent made giving bit power quite surprised find vast majority american think woman represent worst modern america ego control"
two = "I love how racisim is rising "
three = "	Yes indeed. She sort of reminds me of the elder lady that played the part in the movie Titanic who was telling her story!!! And I wouldn't have wanted to cover who I really am!! I would be proud!!!! WE should be proud of our race no matter what it is!!"
four = "@ghostofthor same energy as the YOU DONT LOOK INDIAN STOP LYING ABOUT BEING SOUTH ASIAN cc	"
five = "Get rid of them and their anchor babies and grand babies. Fucking worthless guinees "
#not_hate
six = "I'm pretty sure that Atticus is going to die in the sequel."
seven = "I don't know why anyone really gives a shit about whether Atticus is racist or not. While racism sucks, the fact remains--there are racist people in the world. I hope our society doesn't become so politically correct that even authors have to fear publishing a book with a certain character for fear of being deemed racist for doing so. Writers write about people. The public--well, we bitch if a piece isn't realistic enough and we bitch if it's too realistic. How can anyone win like this?"
eight = "He is afghan.  Must be muslim and future enemy? Why caring about it?"
nine = "The little kid is so cute!"

#hate

ten = "cops like these are the biggest pussies. as a veteran, i assure you it is easy to detain a person without using much force, ESPECIALLY if you have a fucking squad against a single person. I'd like to take one of these guys out back and 'watch him trip and fall'"
eleven = "Pigs can do this to kids in schools, but PARENTS can beat their  asses at HOME????? Wtf?????"
twelve = "Fuck u AJ+ we know it's easy to follow Gobbel's law, keep repeating lies on and on and You start feeling it as true, those Hindu pilgrims didn't die of accident in Godhra train massacre, it was a planned conspiracy in which local Muslim leaders having allegiance to Congress party were found to be involved, the key conspirator being arrested just recently, having been absconding for the past 15 years, so check the facts before lying blatantly!!!"






text = six

print(text)
text = preprocess_text(text)
print(text)

inputs = tokenize_text(text)
outputs = multilabel_model(inputs['input_ids'])

print("Logits:",outputs)
probabilities = F.softmax(outputs, dim=1)
print("Probs:", probabilities)
print("Output:  ", columns_to_transform[torch.argmax(probabilities).item()])






### Her ends the Preprocessing
