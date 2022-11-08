import pandas as pd
import numpy as np
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.base import BaseEstimator

import tensorflow as tf
from keras.models import load_model, Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import pad_sequences

from prepro import pre_processing, keyword_filter, word_embedding

class Attribute17(BaseEstimator):
    def __init__(self):
        self.lr = pickle.load(open('lr_17_model.sav', 'rb'))
        self.lstm = load_model('lstm_17_model.h5')
        self.ada = pickle.load(open('ada_17_model.sav', 'rb'))
        self.tok = pickle.load(open('tok_17_model.sav', 'rb'))

    def predict(self, df):
        df = keyword_filter(df, ['compensation', 'remuneration'])
        df['preprocessed'] = df['sentence'].apply(lambda x: pre_processing(x))
        if df.empty:
            return df
        X = word_embedding(df, 'preprocessed', 17)
        
        lr_model = pickle.load(open('lr_17_model.sav', 'rb'))
        ada_model = pickle.load(open('ada_17_model.sav', 'rb'))
        
        
        ada_pred = ada_model.predict(X)
        
        df['flag'] = ada_pred
        
        ## Returns 1s only
        df_ones = df[df['flag'] == 1]
        
        df_ones = df_ones[['sentence', 'flag']]
        return df_ones
