from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle

import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

from pipeline.prepro import stemming, lemmatization, remove_stop_words

class Attribute16:
    def __init__(self, threshold=0.5):
        train_data = pd.read_csv('pipeline/transition_data.csv')
        self.X = train_data[train_data['have_transition_plan']]['corpus']
        self.X = lemmatization(stemming(remove_stop_words(self.X)))
        self.vectorizer = TfidfVectorizer()
        self.tf_idf_matrix = self.vectorizer.fit_transform(self.X)
        self.threshold = threshold

    def process(self, X):
        return lemmatization(stemming(remove_stop_words(X)))

    def predict(self, df, column='sentence'):
        X = df[column]
        X = self.process(X)
        X_vectorized = self.vectorizer.transform(X)
        predict = []
        cosine = cosine_similarity(X_vectorized, self.tf_idf_matrix)
        for i in range(len(X)):
            input_cosine = cosine[i]
            predict.append(max(input_cosine) >= self.threshold)
        df_res = df.copy()        
        df_res['flag'] = list(map(int, predict))
        df_ones = df_res[df_res['flag'] == 1]
        df_ones = df_ones[[column]]
        return df_ones
