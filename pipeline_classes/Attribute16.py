from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle

class Attribute16:
    def __init__(self):
        self.vectorizer = pickle.load(open('tfidf_16_model.pkl', 'rb'))

    def fit(self, X, y=None):
        return self

    def predict(self, df, column='sentence'):
        X = df[column]
        X_vectorized = self.vectorizer.transform(X)
        for i in range(self.tf_idf_matrix.shape[0]):
            cosine = cosine_similarity(self.tf_idf_matrix[i], input_vec)[0]
            angle_list = np.rad2deg(np.arccos(cosine))
            predict.append(min(angle_list) <= self.threshold)
        df['flag'] = predict.astype(int)
        return df
