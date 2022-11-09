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

from prepro import pre_processing, keyword_filter, is_quantitative

class Attribute25:
    def __init__(self):
        self.relevance_vectorizer = pickle.load(open('models/relevance_vectorizer.pkl', 'rb'))
        self.relevance_model = pickle.load(open('models/relevance_rf.pkl', 'rb'))
        self.scale_vectorizer = pickle.load(open('models/scale_vectorizer.pkl', 'rb'))
        self.scale_model = load_model('models/scale_nn.h5')        

    def pred_helper(self, X, vectorizer, model, pred_type):
        X_vec = pd.DataFrame(vectorizer.transform(X['preprocessed']).todense(), columns=vectorizer.get_feature_names_out())
        if pred_type == 'relevance':
            y_pred = model.predict(X_vec)
        elif pred_type == 'scale':
            y_pred = [i+1 for i in np.argmax(model.predict(X_vec, verbose=0), axis=1)]
        df = pd.DataFrame({'sentence': X['sentence'],'preprocessed': X['preprocessed'], 'pred_label': y_pred})
        return df

    def predict(self, df):
        df_filtered = keyword_filter(df, ['supplier', 'supply chain', 'value chain'], column='sentence')
        if df_filtered.empty:
            return df_filtered
        df_filtered['preprocessed'] = df_filtered['sentence'].apply(lambda x: pre_processing(x))

        # predict
        relevance = self.pred_helper(df_filtered, self.relevance_vectorizer, self.relevance_model, 'relevance')
        scale = self.pred_helper(relevance[relevance['pred_label'] == True], self.scale_vectorizer, self.scale_model, 'scale')

        # get final results
        relevance['quantitative'] = relevance['sentence'].apply(lambda x: is_quantitative(x))
        relevant = pd.DataFrame(relevance[relevance['pred_label'] == True]['sentence'])
        attribute_23 = pd.DataFrame(relevance[(relevance['quantitative'] == True) & (relevance['pred_label'] == True)]['sentence'])
        attribute_25 = list(set(scale['pred_label']))

        print('# Relevant sentences found: ' + str(relevant.shape[0]))

        return attribute_25, relevant
