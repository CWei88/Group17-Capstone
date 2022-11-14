import pandas as pd
import numpy as np
import re
import tensorflow as tf

import warnings
warnings.filterwarnings("ignore")

## Nltk imports
import nltk
from nltk.stem.snowball import SnowballStemmer
nltk.download('stopwords')

# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

import pickle 

from pipeline.prepro import pre_processing, keyword_filter, stop_words_removal, porter_stemmer, custom_standardization

class Attribute8:
    '''
    This class utilises pretrained models to generate sentences related to
    attribute 8:
    
    Do you have an active program to support increasing green space and promote biodiversity?
    '''

    def __init__(self):
        '''
        Initialization for Attribute8, which are used to load models and stopwords
        used for text classification.
        '''
        self.model = pickle.load(open('pipeline/models/model_8.pkl', 'rb'))
        self.vectorizer = pickle.load(open('pipeline/models/vectorizer_8.pkl', 'rb'))
        self.stop_words = stopwords.words('english')

    def predict(self, df, column='sentence'):
        '''
        Main Prediction function for attribute 8.

        Parameters
        -----------
        df: pandas Dataframe
            The dataframe that will be used for text classification.

        column: string
            The column that the sentences generated from the pdf is stored in.
            By default, the column is named sentence.

        Return
        ------
        label: string
            The label to indicate if any relevant sentences are found.
            If no sentences are found, No is returned. If relevant sentences are found, it will return Yes.

        sentences: pandas Dataframe
            The dataframe of sentences that are relevant to attribute 8.

        '''
        ## Preprocessing
        df = keyword_filter(df,['biodiversity','green space','program','animal','fish','bird','avian','tree','forest','coastal','beach','shoreline',
                                         'clean-up','specie','ecosystem','system','project','protection','conservation','natural resources','wildlife','habitat'])
        if df.empty:
            return df
        df = df.reset_index()
        df['sentence1'] = df[column].apply(lambda x:tf.compat.as_str_any(custom_standardization(x).numpy()))
        df['sentence1'] = df['sentence1'].apply(lambda x:stop_words_removal(str(x), self.stop_words))
        df['sentence1'] = df['sentence1'].apply(lambda x:porter_stemmer(str(x)))
        X = self.vectorizer.transform(df['sentence'])

        ## Predicting models.
        res = self.model.predict(X)
        sentences = df[res == 1]
        if len(sentences)>0:
            label = 'Yes'
        else:
            label = 'No'
        return label, sentences
