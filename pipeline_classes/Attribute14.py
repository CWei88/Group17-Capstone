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
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('omw-1.4')

import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from imblearn.over_sampling import RandomOverSampler

import spacy
from spacy.matcher import Matcher 
from spacy.tokens import Span 
nlp = spacy.load('en_core_web_sm')

from prepro import pre_processing, keyword_filter, word_embedding

class Attribute14:

    def __init__(self):
        self.lr_model = pickle.load(open('lr_14_model.sav', 'rb'))
        self.rf_model = pickle.load(open('rf_14_model.sav', 'rb'))
        self.ada_model = pickle.load(open('ada_14_model.sav', 'rb'))

    
    def predict(self, df):
        df = preprocessing(df)
        df = keyword_filter(df, ['ghg', 'sbti', 'tcfd', 'sasb', r'scope /d'])
        X = word_embedding(df, 'words', 14)


        lr_pred = self.lr_model.predict(X)
        rf_pred = self.rf_model.predict(X)
        ada_pred = self.ada_model.predict(X)

        ## Ensemble voting
        df_combi = pd.DataFrame([lr_pred, rf_pred, ada_pred]).transpose()
        df_combi['total'] = df_combi.mode(axis=1)[0]
        df = df.reset_index()
        df['flag'] = df_combi['total']
        print(df)

        ### return 1s only
        df_ones = df[df['flag'] == 1]

        for index, rows in df_ones.iterrows():
            res = []
            if ('ghg' in rows['words'].lower()) or (r'scope \d' in rows['words'].lower()):
                res.append('GHG')
            if ('sbti' in rows['words'].lower()) or ('science based targets' in rows['words'].lower()):
                res.append('SBTi')
            if ('tcfd' in rows['words'].lower()) or ('climate-related financial disclosures' in rows['words'].lower()):
                res.append('TCFD')
            if ('sasb' in rows['words'].lower()) or ('sustainability accounting' in rows['words'].lower()):
                res.append('SASB')

            df_ones.at[index, 'methodologies'] = str(res)
        df_ones = df_ones[['words', 'methodologies', 'flag']]
        return df_ones
