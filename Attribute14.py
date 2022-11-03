### Imports ###

from scipy import spatial
import math
import os
import json
import string

import warnings
warnings.filterwarnings("ignore")

# PDF text extraction
from pdfminer3.layout import LAParams, LTTextBox
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager
from pdfminer3.pdfinterp import PDFPageInterpreter
from pdfminer3.converter import PDFPageAggregator
from pdfminer3.converter import TextConverter

# Others
import string
import re
from tqdm.notebook import tqdm

# Text pre-processing (Tokenization, Stemming, Lemmatization)
import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('omw-1.4')

#Gensim stopwords
import gensim
from gensim.parsing.preprocessing import remove_stopwords
stopwords = gensim.parsing.preprocessing.STOPWORDS

import seaborn as sns
import matplotlib.pyplot as plt 
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import plot_confusion_matrix, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import pad_sequences

from imblearn.over_sampling import RandomOverSampler

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

import spacy
from spacy.matcher import Matcher 
from spacy.tokens import Span 
nlp = spacy.load('en_core_web_sm')

from keybert import KeyBERT

from preprocessing import preprocessing, keyword_filter, word_embedding

class Attribute14(dataiku.apinode.predict.predictor.ClassificationPredictor):

    
    def predict(self, df):
        df = preprocessing(df)
        df = keyword_filter(df, ['ghg', 'sbti', 'tcfd', 'sasb', r'scope /d'])
        X = word_embedding(df, 'words', 14)
        lr_model = pickle.load(open('lr_14_model.sav', 'rb'))
        rf_model = pickle.load(open('rf_14_model.sav', 'rb'))
        ada_model = pickle.load(open('ada_14_model.sav', 'rb'))

        lr_pred = lr_model.predict(X)
        rf_pred = rf_model.predict(X)
        ada_pred = ada_model.predict(X)

        ## Ensemble voting
        df_combi = pd.DataFrame([lr_pred, rf_pred, ada_pred]).transpose()
        df_combi['total'] = df_combi.mode(axis=1)[0]
        df = df.reset_index()
        df['flag'] = df_combi['total']
        print(df)

        ### return 1s only
        df_ones = df[df['flag'] == 1]

        for index, rows in df_ones.iterrows():
            res = []
            if ('ghg' in rows['words'].lower()) or (r'scope \d' in rows['words'].lower()):
                res.append('GHG')
            if ('sbti' in rows['words'].lower()) or ('science based targets' in rows['words'].lower()):
                res.append('SBTi')
            if ('tcfd' in rows['words'].lower()) or ('climate-related financial disclosures' in rows['words'].lower()):
                res.append('TCFD')
            if ('sasb' in rows['words'].lower()) or ('sustainability accounting' in rows['words'].lower()):
                res.append('SASB')

            df_ones.at[index, 'methodologies'] = str(res)
        df_ones = df_ones[['words', 'methodologies', 'flag']]
        return df_ones
