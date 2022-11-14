import math
import numpy as np
import pandas as pd
import pickle

from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

from pipeline.prepro import pre_processing, keyword_filter, word_embedding, qa_filtering

class Attribute15:
    '''
    Class containing functions used to perform text classification on Attribute 7 and 15:

    Have your Scope 1 - 2 & Scope 3 emissions been verified by a third party?
    Are your emission reduction targets externally verified/assured? 
    '''

    def __init__(self, bert_model='pipeline/bert_model'):
        '''
        Initialization function for Attribute15.
        It loads the pretrained mdoels for Attribute15, as well as the bert_model to be used
        for BERTQA to identify companies verifiying emissions and emissions target.

        Parameters
        ----------
        bert_model: str
            The bert_model to be used for BERTQA. By default, it assumes that the bert_model
            has been preinstalled in the bert_model folder, and will be using the folder
            to pretrain the bert_model to identify companies performing external verification.
        '''
        self.ada = pickle.load(open('pipeline/models/ada_15_model.pkl', 'rb'))
        self.svc = pickle.load(open('pipeline/models/svc_15.pkl', 'rb'))
        self.tfidf_2 = pickle.load(open('pipeline/models/tfidf_15_2.pkl', 'rb'))
        self.bert_model = bert_model

    def predict(self, df, further_precision=False):
        '''
        Prediction function to predict if a sentence is relevant to attribute 7 and 15,
        using keyword filtering, lemmatization and pretrained models.

        Parameters
        ----------
        df: pandas DataFrame
            The dataframe of sentences that will be used for text classification
            and prediction to generate relevant sentences.

        further_precision: Boolean
            If True, it will perform further text classification to identify sentences
            that names a company that has verified emissions and emissions target.
            However, current implementation is not as accurate due to a lack of training data
            regarding company-specific verification.

            If no company-specific verification is found, it will use the original text classification
            to generate outputs.

        Result
        ------
        df_ones: pandas DataFrame
            The resultant dataframe containing the relevant sentences identified,
            as well as the companies that have audited the emissions and targets.
        '''
        ## Preprocessing
        df = keyword_filter(df, ['assurance', 'limited assurance', 'externally verified', 'independent', 'third-party'])
        df['preprocessed'] = df['sentence'].apply(lambda x: pre_processing(x))
        if df.empty:
            return df
        X = word_embedding(df, 'preprocessed', 15)
        
        ada_pred = self.ada.predict(X)
        
        ##return 1s only
        df['flag'] = ada_pred
        df_ones = df[df['flag'] == 1]

        ## Further filtering for companies
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

            ## If further filtering does not return an empty dataframe
            if not df_verified.empty:
                res = qa_filtering(df_verified, self.bert_model)
                df_verified['auditors'] = res
                df_verified = df_verified[['sentence', 'auditors', 'further_flag']]
                return df_verified
            else:
                print("Unable to conduct further separation. Original separation will be used instead.")

        ## This is used if further filtering is empty, or no further filtering is used.
        res = qa_filtering(df_ones, self.bert_model)
        df_ones['auditors'] = res

        df_ones = df_ones[['sentence', 'auditors', 'flag']]
        return df_ones
    
