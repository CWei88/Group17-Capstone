### Imports ###

from scipy import spatial
import math
import os
import json
import string

import warnings
warnings.filterwarnings("ignore")

# PDF text extraction
from pdfminer3.layout import LAParams, LTTextBox
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager
from pdfminer3.pdfinterp import PDFPageInterpreter
from pdfminer3.converter import PDFPageAggregator
from pdfminer3.converter import TextConverter

# Others
import string
import re
from pprint import pprint
from tqdm.notebook import tqdm
import io

# Text pre-processing (Tokenization, Stemming, Lemmatization)
import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('omw-1.4')

# Pdf Extraction Model
import spacy
spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm", disable=['ner'])

#Gensim stopwords
import gensim
from gensim.parsing.preprocessing import remove_stopwords
stopwords = gensim.parsing.preprocessing.STOPWORDS

import numpy as np
import pandas as pd
import PyPDF2
import tabula
from tabula import read_pdf
from tabulate import tabulate
import io
import camelot
from functools import reduce
from pdfminer.high_level import extract_text
import pdf2image

import seaborn as sns
import matplotlib.pyplot as plt 
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import plot_confusion_matrix, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import pad_sequences

from imblearn.over_sampling import RandomOverSampler

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

import spacy
from spacy.matcher import Matcher 
from spacy.tokens import Span 
nlp = spacy.load('en_core_web_sm')

from keybert import KeyBERT

class Preprocessing:
    
    def preprocessing(self, df):
        df = df.drop_duplicates()
        bert=KeyBERT()
        kw = []
        for i in tqdm(df['words']):
            kw.append(bert.extract_keywords(i, keyphrase_ngram_range=(2, 2), stop_words='english'))
        df['kw'] = kw
        return df
    
    def keyword_filter(self, df):
        def func(kw, key):
            if any(any(w in word[0] for w in key) for word in kw):
                return True
    
        df_filtered = df[df['kw'].apply(lambda x: func(x, keywords)) == True]
        return df_filtered
    
    def word_embedding(df, embed_column, attribute_no, embedding_model='tfidf'):
        if embedding_model == 'tfidf': ##save fit model and transform here
            X = df[embed_column]
            X = X.apply(lambda x: x.lower())
            if attribute_no == 14:
                tfidf = pickle.load(open('tfidf_14_model.sav', 'rb'))
            elif (attribute_no == 7) or (attribute_no == 15):
                tfidf = pickle.load(open('tfidf_15_model.sav', 'rb'))
            elif attribute_no == 17:
                tfidf = pickle.load(open('tfidf_17_model.sav', 'rb'))
            else:
                raise Exception(f"Wrong Model used for attribute: {attribute_no}")
            x = tfidf.transform(X)
            X_encoded = pd.DataFrame(x.toarray())
            return X_encoded
        else:
            raise Exception("No model found")
            
    def qa_filtering(df):
        model_name = "deepset/roberta-base-squad2"
        nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

        res = []
        q1 = 'Who audited the targets?'
        q2 = 'Who assured the targets?'
        q3 = 'Who verified the targets?'
        for i in df['words']:
            QA_1 = {
                'question': q1,
                'context': i
            }
            QA_2 = {
                'question': q2,
                'context': i
            }
            QA_3 = {
                'question': q3,
                'context': i
            }

            ans1 = nlp(QA_1)['answer']
            score1 = nlp(QA_1)['score']
            ans2 = nlp(QA_2)['answer']
            score2 = nlp(QA_2)['score']
            ans3 = nlp(QA_3)['answer']
            score3 = nlp(QA_3)['score']

            maxi = max([score1, score2, score3])
            if maxi == score1:
                res.append(ans1)
            elif maxi == score2:
                res.append(ans2)
            else:
                res.append(ans3)
        return res
