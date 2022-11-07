from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle

class Attribute16:
    def __init__(self, threshold=60):
        train_data = pd.read_csv('transition_data.csv')
        self.X = train_data[train_data['have_transition_plan']]['corpus']
        self.vectorizer = TfidfVectorizer()
        self.tf_idf_matrix = self.vectorizer.fit_transform(self.X)
        self.threshold = threshold

    def predict(self, df, column='sentence'):
        X = df[column]
        X_vectorized = self.vectorizer.transform(X)
        predict = []
        for i in range(self.tf_idf_matrix.shape[0]):
            cosine = cosine_similarity(self.tf_idf_matrix[i], X_vectorized)[0]
            angle_list = np.rad2deg(np.arccos(cosine))
            predict.append(min(angle_list) <= self.threshold)
        df_res = df.copy()        
        df_res['flag'] = list(map(int, predict))
        return df_res
