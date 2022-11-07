import math
import numpy as np
import pandas as pd
import pickle

from sklearn.linear_model import LinearRegression
from prepro import pre_processing, keyword_filter, word_embedding, qa_filtering

class Attribute15:

    def __init__(self):
        self.lr_model = pickle.load(open('lr_15_model.sav', 'rb'))

    def predict(self, df):
        df = preprocessing(self.df)
        df = keyword_filer(df, ['assurance', 'limited assurance',
                                'externally verified', 'independent', 'third-party'])
        X = word_embedding(df, 'words', 15)

        lr_pred = self.lr_model.predict(X)

        ## Return ones only
        df['flag'] = lr_pred
        df_ones = df[df['flag'] == 1]
        res = qa_filtering(df_ones)
        df_ones['auditors'] = res

        df_ones = df_ones[['words', 'auditors', 'flag']]
        return df_ones
    
