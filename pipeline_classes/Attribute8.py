######################################### IMPORTING PACAKGES #############################
from scipy import spatial
import pandas as pd
import os
import json
import numpy as np
import string
import warnings
warnings.filterwarnings("ignore")
import sys  
import os
from dateutil.parser import parse
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# Others
import string
import re
import io

import nltk
from nltk.stem.snowball import SnowballStemmer

# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
import spacy
nlp = spacy.load("en_core_web_sm", disable=['ner'])

from sklearn.feature_extraction import text
stop_words = text.ENGLISH_STOP_WORDS.union(stop_words)
import pickle 
import tensorflow as tf
from prepro import pre_processing, keyword_filter, stop_words_removal, porter_stemmer, custom_standardization

class Attribute8:

    def __init__(self):
        self.model = pickle.load(open('model_8.pkl', 'rb'))
        self.vectorizer = pickle.load(open('vectorizer_8.pkl', 'rb'))
        

    def predict(self, df):
        df = keyword_filter(df,['net-zero','net-zero','carbon neutral','commitment','target','long term','2030','2040','2045','2050',' neutrality','carbon free','carbon-free','zero emission','zero GHG emission','zero CO2 emission','SBTi','Science Based Targets initiative'])
        df = df.reset_index()
        df['sentence1'] = df['sentence'].apply(lambda x:tf.compat.as_str_any(custom_standardization(x).numpy()))
        df['sentence1'] = df['sentence1'].apply(lambda x:stop_words_removal(str(x)))
        df['sentence1'] = df['sentence1'].apply(lambda x:porter_stemmer(str(x)))
        X = self.vectorizer.transform(df['sentence'])
        res = self.model.predict(X)
        sentences = list(df[res == 1]['sentence'])
        if len(sentences)>0:
            label = 'Yes'
        else:
            label = 'False'
        return label, sentences
