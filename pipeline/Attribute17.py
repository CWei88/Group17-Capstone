import pandas as pd
import numpy as np
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostClassifier

import tensorflow as tf
from keras.models import load_model, Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import pad_sequences

from pipeline.prepro import pre_processing, keyword_filter, word_embedding

class Attribute17:
    '''
    Class containing functions used to perform text classification on Attribute 17:

    Do you provide incentives to your senior leadership team for the management of climate related issues?
    '''
    def __init__(self):
        '''
        Initialization function for Attribute 17.
        It loads pretrained models to be used to predict whether a sentence is relevant.
        '''
        self.ada = pickle.load(open('pipeline/models/ada_17_model.sav', 'rb'))

    def predict(self, df):
        '''
        Prediction function to predict if a sentence is relevant to attribute 17,
        using keyword filtering, lemmatization and pretrained models.

        Parameters
        ----------
        df: pandas DataFrame
            The dataframe of sentences that will be used for text classification
            and prediction to generate relevant sentences.

        Result
        ------
        df_ones: pandas DataFrame
            The resultant dataframe containing the relevant sentences identified
            in the text classification process.
        '''
        ## Preprocessing
        df = keyword_filter(df, ['compensation', 'remuneration'])
        df['preprocessed'] = df['sentence'].apply(lambda x: pre_processing(x))
        if df.empty:
            return df
        X = word_embedding(df, 'preprocessed', 17)

        ## Prediction process
        ada_pred = self.ada.predict(X)
        
        df['flag'] = ada_pred
        
        ## Returns 1s only
        df_ones = df[df['flag'] == 1]
        
        df_ones = df_ones[['sentence', 'flag']]
        return df_ones
