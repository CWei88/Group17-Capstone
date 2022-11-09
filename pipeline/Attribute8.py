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
import tensorflow as tf

import nltk
from nltk.stem.snowball import SnowballStemmer

# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
import en_core_web_sm
nlp = en_core_web_sm.load(disable=['ner'])

from sklearn.feature_extraction import text
stop_words = text.ENGLISH_STOP_WORDS.union(stop_words)
import pickle 

from prepro import pre_processing, keyword_filter, stop_words_removal, porter_stemmer, custom_standardization

class Attribute8:

    def __init__(self):
        self.model = pickle.load(open('models/model_8.pkl', 'rb'))
        self.vectorizer = pickle.load(open('models/vectorizer_8.pkl', 'rb'))
        self.stop_words = text.ENGLISH_STOP_WORDS.union(stop_words)

    def predict(self, df, column='sentence'):
        df = keyword_filter(df,['biodiversity','green space','program','animal','fish','bird','avian','tree','forest','coastal','beach','shoreline',
                                         'clean-up','specie','ecosystem','system','project','protection','conservation','natural resources','wildlife','habitat'])
        if df.empty:
            return df
        df = df.reset_index()
        df['sentence1'] = df['sentence'].apply(lambda x:tf.compat.as_str_any(custom_standardization(x).numpy()))
        df['sentence1'] = df['sentence1'].apply(lambda x:stop_words_removal(str(x), stop_words))
        df['sentence1'] = df['sentence1'].apply(lambda x:porter_stemmer(str(x)))
        X = self.vectorizer.transform(df['sentence'])
        res = self.model.predict(X)
        sentences = df[res == 1]
        if len(sentences)>0:
            label = 'Yes'
        else:
            label = 'False'
        return label, sentences
