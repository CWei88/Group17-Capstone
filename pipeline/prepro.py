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
import io

# Text pre-processing (Tokenization, Stemming, Lemmatization)
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import PorterStemmer, LancasterStemmer, WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer, RegexpTokenizer, sent_tokenize, word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('omw-1.4')

# Pdf Extraction Model
import en_core_web_sm
nlp = en_core_web_sm.load()

#Gensim stopwords
import gensim
from gensim.parsing.preprocessing import remove_stopwords
stopwords = gensim.parsing.preprocessing.STOPWORDS

import numpy as np
import pandas as pd
import PyPDF2
import tabula
from tabula import read_pdf
import pdf2image

import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

import tensorflow as tf
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import pad_sequences

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

import spacy
from spacy.matcher import Matcher 
from spacy.tokens import Span 

def clean(line):
    line = re.sub(r'[0-9\.]+', '', line) # remove digits
    line = re.sub(r'[^\w\s]','', line) # remove punctuation
    return line

def stemming(line):
    stemmer = SnowballStemmer(language='english')
    return [stemmer.stem(token) for token in line]

def lemmatization(line):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in line]

def remove_stop_words(line):
    return [remove_stopwords(token) for token in line]

def pre_processing(line):
    tokenizer = TreebankWordTokenizer()

    tokenized_line = tokenizer.tokenize(clean(line))
    preprocessed_line = stemming(lemmatization(remove_stop_words(tokenized_line)))
        
    return ' '.join([token for token in preprocessed_line if token != ''])

    
def keyword_filter(df, keywords, column='sentence'):
    filtered = []
    for s in np.array(df[column]):
        sentence = s.lower()
        for k in keywords:
            if k in sentence:
                filtered.append([s, k])
        
    filtered_df = pd.DataFrame(filtered, columns=['sentence', 'keyword(s)']).groupby(['sentence']).agg({'keyword(s)': lambda x: list(x.unique())}).reset_index()
    return filtered_df
    
def word_embedding(df, embed_column, attribute_no, embedding_model='tfidf'):
    if embedding_model == 'tfidf': ##save fit model and transform here
        X = df[embed_column]
        X = X.apply(lambda x: x.lower())
        if attribute_no == 14:
            tfidf = pickle.load(open('models/tfidf_14_model.sav', 'rb'))
        elif (attribute_no == 7) or (attribute_no == 15):
            tfidf = pickle.load(open('models/tfidf_15_model.sav', 'rb'))
        elif attribute_no == 17:
            tfidf = pickle.load(open('models/tfidf_17_model.sav', 'rb'))
        else:
            raise Exception(f"Wrong Model used for attribute: {attribute_no}")
        x = tfidf.transform(X)
        X_encoded = pd.DataFrame(x.toarray())
        return X_encoded
    else:
        raise Exception("No model found")
        
def qa_filtering(df):
    model_name = "bert_model"
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

    res = []
    q1 = 'Who audited the targets?'
    q2 = 'Who assured the targets?'
    q3 = 'Who verified the targets?'
    for i in df['sentence']:
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

def is_quantitative(x):
    x = x.lower()

    x = re.sub("[2][0][0-5][0-9]", "", x) #remove years
    x = re.sub("fy[0-9]+", "", x) #remove numbers that represent financial year e.g. FY21
    x = re.sub("tier\s*[0-9]", "", x) #remove numbers related to tiers
    x = re.sub("scope\s*[0-9]", "", x) #remove numbers related to scope
    x = re.sub("co2", "", x) #remove 'CO2'
    x = re.sub("cid.+", "", x) #remove 'cid'
    x = re.sub("[0-9]+[:)]|[#]+[0-9]", "", x) #remove numbers for indexing e.g. 1) or #1 or 1:

    return re.search("supplier", x) and len(re.findall(r'\d+', x)) > 0

def stop_words_removal(sentence, stop_words):
    words = sentence.split()
    removed_sentence=[]
    for r in words:
        if not r in stop_words:
            removed_sentence.append(r)
            removed_sentence.append(" ")
    return "".join(removed_sentence)

def porter_stemmer(sentence):
    porter = PorterStemmer()
    token_words=word_tokenize(sentence)
    token_words
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)

def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  return tf.strings.regex_replace(lowercase,
                                  '[%s]' % re.escape(string.punctuation), '')
