### Imports ###

import math
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

# Others
import string
import re

# Text pre-processing (Tokenization, Stemming, Lemmatization)
import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from imblearn.over_sampling import RandomOverSampler

import spacy
from spacy.matcher import Matcher 
from spacy.tokens import Span 
import en_core_web_sm
nlp = en_core_web_sm.load()

from prepro import pre_processing, keyword_filter, word_embedding

class Attribute14():

    def __init__(self):
        self.lr_model = pickle.load(open('models/lr_14_model.sav', 'rb'))
        self.rf_model = pickle.load(open('models/rf_14_model.sav', 'rb'))
        self.svc_model = pickle.load(open('models/svc_14_model.pkl', 'rb'))

    
    def predict(self, df):
        df = keyword_filter(df, ['ghg', 'sbti', 'tcfd', 'sasb', r'scope /d'])
        df['preprocessed'] = df['sentence'].apply(lambda x: pre_processing(x))
        if df.empty:
            return df
        X = word_embedding(df, 'preprocessed', 14)

        lr_pred = self.lr_model.predict(X)
        rf_pred = self.rf_model.predict(X)
        svc_pred = self.svc_model.predict(X)
       
        ## Ensemble voting
        df_combi = pd.DataFrame([lr_pred, rf_pred, svc_pred]).transpose()
        df_combi['total'] = df_combi.mode(axis=1)[0]
        df = df.reset_index()
        df['flag'] = df_combi['total']
        
        ### return 1s only
        df_ones = df[df['flag'] == 1]
        
        for index, rows in df_ones.iterrows():
            res = []
            if ('ghg' in rows['sentence'].lower()) or (r'scope \d' in rows['sentence'].lower()):
                res.append('GHG')
            if ('sbti' in rows['sentence'].lower()) or ('science based targets' in rows['sentence'].lower()):
                res.append('SBTi')
            if ('tcfd' in rows['sentence'].lower()) or ('climate-related financial disclosures' in rows['sentence'].lower()):
                res.append('TCFD')
            if ('sasb' in rows['sentence'].lower()) or ('sustainability accounting' in rows['sentence'].lower()):
                res.append('SASB')
        
            df_ones.at[index, 'methodologies'] = str(res)
        df_ones = df_ones[['sentence', 'methodologies', 'flag']]
        return df_ones
