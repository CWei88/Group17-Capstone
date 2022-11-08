from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix 


class TransitionPlanModel(BaseEstimator):
    def __init__(self, threshold):
        self.threshold = threshold
        self.classifier = TfidfVectorizer()

    def fit(self, X, y=None):
        self.tf_idf_matrix = self.classifier.fit_transform(X)
        return self
    
    def predict(self, X):
        # X is a 2D array for corpus
        input_vec = self.classifier.transform(X)
        predict = [] #2D Boolean Prediction Array
        cosine = cosine_similarity(input_vec, self.tf_idf_matrix)
        for i in range(len(X)):
            input_cosine = cosine[i]
            # angle_list = np.rad2deg(np.arccos(input_cosine))
            predict.append(max(input_cosine) >= self.threshold)
        return predict



transition_df = pd.read_csv('transition_data.csv')
X_fit = transition_df[transition_df["have_transition_plan"]]["corpus"]
X = transition_df["corpus"]
y = transition_df["have_transition_plan"]
X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

transitionModel = TransitionPlanModel(threshold = 0.5)
transitionModel.fit(X_fit[:500])
df = pd.read_csv('citycon.csv')
X_t = df['sentence']
print(transitionModel.predict(X_t))
y_pred = transitionModel.predict(X_test)
confusion = confusion_matrix(y_test,y_pred)
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]
print(f"Dataset Size: {transition_df.shape[0]}")
print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")
print(f"Precision: {metrics.precision_score(y_test, y_pred)}")
print(f"Recall: {metrics.recall_score(y_test, y_pred)}")
print(f"F1 Score: {metrics.f1_score(y_test, y_pred)}")
