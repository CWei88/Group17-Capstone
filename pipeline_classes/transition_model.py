from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix

import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer 
nltk.download('wordnet')
import gensim
from gensim.parsing.preprocessing import remove_stopwords
stopwords = gensim.parsing.preprocessing.STOPWORDS

def preprocess(corpus):
        def stemming(corpus):
            stemmer = SnowballStemmer(language='english')
            revisions = [stemmer.stem(line) for line in corpus]
            return revisions

        def lemmatization(corpus):
            lemmatizer = WordNetLemmatizer()
            revisions = [lemmatizer.lemmatize(line) for line in corpus]
            return revisions

        def remove_stop_words(corpus):
            revisions = [remove_stopwords(line) for line in corpus]
            return revisions

        return lemmatization(stemming(remove_stop_words(corpus)))

class TransitionPlanModel(BaseEstimator):
    def __init__(self, threshold):
        self.threshold = threshold
        self.classifier = TfidfVectorizer()

    # Fit only correctly identify sentences
    def fit(self, X, y=None):
        X = preprocess(X)
        self.tf_idf_matrix = self.classifier.fit_transform(X)
        return self
    
    def predict(self, X):
        X = preprocess(X)
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
transitionModel.fit(X_fit[:600])
y_pred = transitionModel.predict(X_test)
confusion = confusion_matrix(y_test,y_pred)
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]
print(f"Dataset Size: {transition_df.shape[0]}")
print(f"Fit Data size: {len(X_fit)}")
print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")
print(f"Precision: {metrics.precision_score(y_test, y_pred)}")
print(f"Recall: {metrics.recall_score(y_test, y_pred)}")
print(f"F1 Score: {metrics.f1_score(y_test, y_pred)}")

alstria_corpus = pd.read_csv('alstria.csv')["sentence"]
alstria_pred = transitionModel.predict(alstria_corpus)
print(f"Alstria Prediction Num of True: {sum(alstria_pred)}")
print(f"Alstria Prediction Correct Sentences: {alstria_corpus[alstria_pred]}")

citycon_corpus = pd.read_csv('citycon.csv')["sentence"]
citycon_pred = transitionModel.predict(citycon_corpus)
print(f"Citycon Prediction Num of True: {sum(citycon_pred)}")
print(f"Citycon Prediction Correct Sentences: {citycon_corpus[citycon_pred]}")
