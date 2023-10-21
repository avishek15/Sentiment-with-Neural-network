from typing import Tuple, Any
import nltk
import pandas as pd
import os
from pandas import DataFrame
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import string
import pickle

nltk.download('punkt')
nltk.download('stopwords')


def get_data() -> Tuple[DataFrame | Any, DataFrame | Any, DataFrame | Any]:
    train_data = pd.read_csv("data/Train.csv")
    test_data = pd.read_csv("data/Test.csv")
    valid_data = pd.read_csv("data/Valid.csv")
    return train_data, test_data, valid_data


def count(all_words: list[str], word: str) -> int:
    found_indexes = [idx for idx in range(len(all_words)) if all_words[idx] == word]
    return len(found_indexes)


def preprocess_data(df1: pd.DataFrame) -> dict[str, int]:
    if os.path.exists("encoder.pkl"):
        with open("encoder.pkl", "rb") as f:
            w_to_i = pickle.load(f)
        return w_to_i
    all_sentences = df1['text'].to_list()
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english') + list(string.punctuation))
    all_words = [stemmer.stem(i) for s in all_sentences for i in word_tokenize(s.lower()) if i not in stop_words]
    counter = Counter(all_words)
    valid_words = counter.most_common(999)
    w_to_i = dict()
    for idx, vw in enumerate(valid_words):
        w_to_i[vw[0]] = idx + 1
    w_to_i['<UNK>'] = 0
    with open("encoder.pkl", "wb") as f:
        pickle.dump(w_to_i, f)
    return w_to_i


def encoder(sentence: str, encoding_dict: dict) -> list[int]:
    vector = []
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english') + list(string.punctuation))
    all_words = [stemmer.stem(i) for i in word_tokenize(sentence.lower()) if i not in stop_words]
    for w in all_words:
        if w in encoding_dict.keys():
            vector.append(encoding_dict[w])
        else:
            vector.append(encoding_dict['<UNK>'])
    return vector



