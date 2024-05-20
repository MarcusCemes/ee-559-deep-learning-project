import re

import contractions
import nltk
import torch
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pandas import DataFrame, read_parquet
from torch import tensor
from torch.utils.data import Dataset

TOKENIZER_MAX_LEN = 256

COL_SCORE = "hate_speech_score"

COL_TEXT = "text"

COLS_CLASSES = [
    "respect",
    "insult",
    "humiliate",
    "status",
    "dehumanize",
    "violence",
    "genocide",
    "attack_defend",
]


stopwords_downloaded = False
wordnet_downloaded = False


class HateSpeechDataset(Dataset):
    def __init__(self, dataset, tokenizer, device="cuda"):

        self.tokenizer = tokenizer
        self.dataset = dataset
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        text = str(self.dataset[COL_TEXT].iloc[idx])

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=TOKENIZER_MAX_LEN,
            padding="max_length",
            truncation=True,
            return_token_type_ids=True,
        )

        ids = encoding["input_ids"]
        mask = encoding["attention_mask"]
        token_type_ids = encoding["token_type_ids"]

        target = self.dataset[COL_SCORE].values[idx]
        targets = self.dataset[COLS_CLASSES].values[idx]

        return {
            "input_ids": tensor(ids, dtype=torch.long),
            "attention_mask": tensor(mask, dtype=torch.long),
            "token_type_ids": tensor(token_type_ids, dtype=torch.long),
            "target": tensor(target, dtype=torch.float).to(self.device),
            "labels": tensor(targets, dtype=torch.float).to(self.device),
        }


def load_dataset(path):
    columns = [*COLS_CLASSES, COL_SCORE, COL_TEXT]
    return read_parquet(path)[columns]


# == Text preprocessing == #


def preprocess_dataset(dataset: DataFrame, update_nltk: bool) -> DataFrame:

    # Remove rows with empty text
    dataset = dataset[dataset["text"] != ""]

    text = dataset["text"]

    text = text.apply(lambda x: _remove_html_tags(x))
    text = text.apply(lambda x: _remove_url(x))
    text = text.str.lower()
    text = text.apply(lambda x: _remove_punctuation(x))
    text = text.apply(lambda x: _expand_contractions(x))
    text = text.apply(lambda x: _remove_stopwords(x, update_nltk))
    text = text.apply(lambda x: _lemmatize_words(x, update_nltk))

    dataset["text"] = text

    return dataset


def _remove_html_tags(text):
    clean = re.compile("<.*?>")
    return re.sub(clean, "", text)


## Remove URL
def _remove_url(text):
    return re.sub(r"http\S+", "", text)


## Remove punctuation
def _remove_punctuation(text):
    return re.sub(r"[^\w\s]", "", text)


def _expand_contractions(text):
    return contractions.fix(text)


def _remove_stopwords(text, update_nltk: bool):
    global stopwords_downloaded

    if not stopwords_downloaded and update_nltk:
        stopwords_downloaded = True
        nltk.download("stopwords")

    stop_words = set(stopwords.words("english"))
    return " ".join([word for word in str(text).split() if word not in stop_words])


def _lemmatize_words(text, update_nltk: bool):
    global wordnet_downloaded

    if not wordnet_downloaded and update_nltk:
        wordnet_downloaded = True
        nltk.download("wordnet")

    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])


# == Column transformation == #


def prepare_labels(dataset: DataFrame) -> DataFrame:

    dataset[COL_SCORE] = dataset[COL_SCORE].apply(lambda x: 1 if x >= 0.5 else 0)

    for column in COLS_CLASSES:
        dataset[column] = dataset[column].apply(lambda x: 1 if x > 0 else 0)

    dataset["target"] = dataset[COL_SCORE].values.tolist()
    dataset["labels"] = dataset[COLS_CLASSES].values.tolist()

    return dataset
