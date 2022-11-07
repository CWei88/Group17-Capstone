from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle

class TransitionPlanModel(BaseEstimator):
    def __init__(self, threshold):
        self.threshold = threshold
        self.classifier = TfidfVectorizer()

    def fit(self, X, y=None):
        self.tf_idf_matrix = self.classifier.fit_transform(X)
        pickle.dump(self.classifier, open('tfidf_16_model.pkl', 'wb'))
        pickle.dump(self.tf_idf_matrix, open('tfidf_16_matrix.pkl', 'wb'))
        return self
    
    def predict(self, X):
        # X is a 2D array for corpus
        input_vec = self.classifier.transform(X)
        predict = [] #2D Boolean Prediction Array
        for i in range(self.tf_idf_matrix.shape[0]):
            cosine = cosine_similarity(self.tf_idf_matrix[i], input_vec)[0]
            angle_list = np.rad2deg(np.arccos(cosine))
            predict.append(min(angle_list) <= self.threshold)
        return predict

transition_df = pd.read_csv('transition_data.csv')
X = transition_df[transition_df["have_transition_plan"]]["corpus"]
X_train , X_test = train_test_split(X,test_size=0.2)

transitionModel = TransitionPlanModel(threshold = 60)
transitionModel.fit(X=X)
test = pd.read_csv('ubm.csv')
print(transitionModel.predict(test['sentence']))
