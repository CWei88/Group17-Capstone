### Imports ###
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import re

## Text pre-processing
import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem.snowball import SnowballStemmer

import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

import spacy
import en_core_web_sm

from pipeline.prepro import pre_processing, keyword_filter, word_embedding

class Attribute14():
    '''
    Class containing functions used to perform text classification on Attribute14:

    What scenario has been utilised, and what methodology was applied?
    '''

    def __init__(self):
        '''
        Initialization class for Attribute14.
        It consists of loading the pretrained models that will be used for text
        classification for Attribute14.
        '''
        self.lr_model = pickle.load(open('pipeline/models/lr_14_model.sav', 'rb'))
        self.rf_model = pickle.load(open('pipeline/models/rf_14_model.sav', 'rb'))
        self.svc_model = pickle.load(open('pipeline/models/svc_14_model.pkl', 'rb'))

    
    def predict(self, df):
        '''
        Prediction function to predict if a sentence is relevant to attribute14,
        using keyword filtering, lemmatization and pretrained models.

        Parameters
        ----------
        df: pandas DataFrame
            The dataframe of sentences that will be used for text classification
            and prediction to generate relevant sentences.

        Result
        ------
        df_ones: pandas DataFrame
            The resultant dataframe containing the relevant sentences identified,
            as well as the methodologies that are identified for each sentence.
        '''

        ## Preprocessing
        df = keyword_filter(df, ['ghg', 'sbti', 'tcfd', 'sasb', r'scope /d'])
        df['preprocessed'] = df['sentence'].apply(lambda x: pre_processing(x))
        if df.empty:
            return df
        X = word_embedding(df, 'preprocessed', 14)

        ## Predicting with pretrained models
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

        ## Sentence filtering to know which methodologies is identified for each sentence.
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
        
            df_ones.at[index, 'methodologies'] = ','.join(res)
        df_ones = df_ones[['sentence', 'methodologies', 'flag']]
        return df_ones
