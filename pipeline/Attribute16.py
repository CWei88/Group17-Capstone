import numpy as np
import pandas as pd
import pickle

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

from pipeline.prepro import stemming, lemmatization, remove_stop_words

class Attribute16:
    '''
    Class to start the process for predicting if a sentence is relevant
    to attribute 16:

    Do you have a low carbon transition plan?
    '''
    def __init__(self, threshold=0.5):
        '''
        Initialization to predict Attribute 16.
        This loads the trained_dataset, as well as perform lemmatization
        and stopwords removal and vectorization using trained model.

        Parameters
        -----------
        threshold: float: (0, 1)
            The threshold used for the cosine similarity model. The higher
            the threshold, the more stringent the requirement for a statement to be
            classified as relevant. By default, the threshold is 0.5

        '''
        train_data = pd.read_csv('pipeline/transition_data.csv')
        self.X = train_data[train_data['have_transition_plan']]['corpus']
        self.X = lemmatization(stemming(remove_stop_words(self.X)))
        self.vectorizer = TfidfVectorizer()
        self.tf_idf_matrix = self.vectorizer.fit_transform(self.X)
        self.threshold = threshold

    def process(self, X):
        '''
        Preprocessing of sentence, involving removing stop words,
        stemming and lemmatization.

        Parameters
        -----------
        X: numpy array
            The array of sentences to be processed.

        Returns
        --------
        X: numpy array
            The preprocessed sentences.
        '''
        return lemmatization(stemming(remove_stop_words(X)))

    def predict(self, df, column='sentence'):
        '''
        The main function to predict whether a sentence is relevant to attribute 16.

        Parameters
        ----------
        df: pandas Dataframe
            The dataframe of sentences that will be used to predict.

        column: str
            The column that will be used to predict Attribute 16 in the dataframe.

        Returns
        -------
        df_ones: pandas Dataframe
            The dataframe that consists of all sentences that are classified to be
            relevant to attribute 16.

        '''
        ## Preprocessing
        X = df[column]
        X = self.process(X)
        X_vectorized = self.vectorizer.transform(X)
        predict = []

        ## Cosine Similarity
        cosine = cosine_similarity(X_vectorized, self.tf_idf_matrix)

        ## Filtering for sentences meeting the threshold.
        for i in range(len(X)):
            input_cosine = cosine[i]
            predict.append(max(input_cosine) >= self.threshold)

        ## Filtering for sentences that are relevant.
        df_res = df.copy()        
        df_res['flag'] = list(map(int, predict))
        df_ones = df_res[df_res['flag'] == 1]
        df_ones = df_ones[[column]]
        return df_ones
