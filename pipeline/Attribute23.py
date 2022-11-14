# General
import pandas as pd
import numpy as np
import re

# Load models
from keras.models import load_model
import pickle

# Text pre-processing
import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer 

import gensim
from gensim.parsing.preprocessing import remove_stopwords
stopwords = gensim.parsing.preprocessing.STOPWORDS

from pipeline.prepro import pre_processing, keyword_filter, is_quantitative

class Attribute23:
    '''
    The class containing functions used to perform text classification for Attribute 23:

    Does your transition plan include direct engagement with suppliers to drive them to reduce their emissions,
    or even switching to suppliers producing low carbon materials?
    '''

    def __init__(self):
        '''
        The initialization function for Attribute 23, consisting of loading pretrained models.
        '''
        self.relevance_vectorizer = pickle.load(open('pipeline/models/relevance_vectorizer.pkl', 'rb'))
        self.relevance_model = pickle.load(open('pipeline/models/relevance_rf.pkl', 'rb'))
        self.scale_vectorizer = pickle.load(open('pipeline/models/scale_vectorizer.pkl', 'rb'))
        self.scale_model = pickle.load(open('pipeline/models/rf_25.pkl', 'rb'))         

    def pred_helper(self, X, vectorizer, model, pred_type):
        '''
        The prediction helper function of Attribute23, which helps to perform text prediction for the model.

        Parameters
        ----------
        X: pandas Dataframe
            The dataframe used to predict the Attribute
        vectorizer: pickle file
            The saved tfidf vectorization model used to process the sentence into vectors.
        model: pickle file
            The corresponding pretrained model used to generate prediction based on the vectorised sentence.
        pred_type: str (DEPRECATED)
            deprecated argument. It was used to indicate the post-processing on prediction results to be used.

        Returns
        -------
        df: pandas DataFrame
            The dataframe containing a list of predicted sentences, as well as the label for each corresponding sentence.
        '''
        X_vec = pd.DataFrame(vectorizer.transform(X['preprocessed']).todense(), columns=vectorizer.get_feature_names_out())
        y_pred = model.predict(X_vec)
        df = pd.DataFrame({'sentence': X['sentence'],'preprocessed': X['preprocessed'], 'pred_label': y_pred})
        return df

    def predict(self, df):
        '''
        The main prediction function for Attribute 23, generating a prediction for the model.
        
        Parameters
        ----------
        df: pandas DataFrame.
            The dataframe of sentences to be used to predict Attribute 23.

        Returns
        -------
        attribute_23: pandas DataFrame.
            The resultant dataframe containing the relevant sentences identified
            in the text classification process for Attribute 23.
        '''
        ## Preprocessing
        df_filtered = keyword_filter(df, ['supplier', 'supply chain', 'value chain'], column='sentence')
        if df_filtered.empty:
            return df_filtered
        df_filtered['preprocessed'] = df_filtered['sentence'].apply(lambda x: pre_processing(x))

        ## Prediction
        relevance = self.pred_helper(df_filtered, self.relevance_vectorizer, self.relevance_model, 'relevance')
        scale = self.pred_helper(relevance[relevance['pred_label'] == True], self.scale_vectorizer, self.scale_model, 'scale')

        ## Post-processing to get final results.
        relevance['quantitative'] = relevance['sentence'].apply(lambda x: is_quantitative(x))
        relevant = pd.DataFrame(relevance[relevance['pred_label'] == True]['sentence'])
        attribute_23 = pd.DataFrame(relevance[(relevance['quantitative'] == True) & (relevance['pred_label'] == True)]['sentence'])
        attribute_25 = list(set(scale['pred_label']))

        print('# Relevant sentences found: ' + str(relevant.shape[0]))

        return attribute_23
