import math
import numpy as np
import pandas as pd
import pickle

from sklearn.linear_model import LinearRegression
from prepro import pre_processing, keyword_filter, word_embedding, qa_filtering

class Attribute15:

    def __init__(self):
        self.ada = pickle.load(open('models/ada_15_model.pkl', 'rb'))
        self.svc = pickle.load(open('mdoels/svc_15.pkl', 'rb'))
        self.tfidf_2 = pickle.load(open('models/tfidf_15_2.pkl', 'rb'))

    def predict(self, df, further_precision=True):
        df = keyword_filter(df, ['assurance', 'limited assurance', 'externally verified', 'independent', 'third-party'])
        df['preprocessed'] = df['sentence'].apply(lambda x: pre_processing(x))
        if df.empty:
            return df
        X = word_embedding(df, 'preprocessed', 15)
        
        ada_pred = self.ada.predict(X)
        
        ##return 1s only
        df['flag'] = ada_pred
        df_ones = df[df['flag'] == 1]
        
        if further_precision:
            new_X = df_ones['preprocessed']
            if new_X.size != 0:
                x = self.tfidf_2.transform(new_X)
                new_test_X = pd.DataFrame(x.toarray())
                sv_pred = self.svc.predict(new_test_X)

                df_ones['further_flag'] = sv_pred
                df_verified = df_ones[df_ones['further_flag'] == 1]
            else:
                df_verified = pd.DataFrame()

            if not df_verified.empty:
                res = qa_filtering(df_verified)
                df_verified['auditors'] = res
                df_verified = df_verified[['sentence', 'auditors', 'further_flag']]
                return df_verified
            else:
                print("Unable to conduct further separation. Original separation will be used instead.")
        
        res = qa_filtering(df_ones)
        df_ones['auditors'] = res

        df_ones = df_ones[['sentence', 'auditors', 'flag']]
        return df_ones
    
