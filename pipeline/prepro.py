### Imports ###
## General Imports
import re
import io
import os
import string
import numpy as np
import pandas as pd
import pickle

import warnings
warnings.filterwarnings("ignore")

## Text pre-processing (Tokenization, Stemming, Lemmatization)
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import PorterStemmer, LancasterStemmer, WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer, RegexpTokenizer, sent_tokenize, word_tokenize

## Spacy Models
import spacy
from spacy.matcher import Matcher 
from spacy.tokens import Span 
import en_core_web_sm
nlp = en_core_web_sm.load()


## Gensim stopwords
import gensim
from gensim.parsing.preprocessing import remove_stopwords
stopwords = gensim.parsing.preprocessing.STOPWORDS

## sklearn models
from sklearn.feature_extraction.text import TfidfVectorizer

## Keras models
import tensorflow as tf
import keras
from keras.preprocessing.text import Tokenizer

## BERTQA
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

def clean(line):
    '''
    Removes digits and punctuation from the sentence.

    Parameters
    ----------
    line: str
        The sentence to be cleaned.

    Returns
    -------
    line: str
        The cleaned sentence.

    '''
    line = re.sub(r'[0-9\.]+', '', line) # remove digits
    line = re.sub(r'[^\w\s]','', line) # remove punctuation
    return line

def stemming(line):
    '''
    Performs stemming to a sentence, such as removing prefixes and suffixes to
    get the original word without any transformation.

    Parameters
    ----------
    line: str
        The sentence to perform stemming.

    Returns
    -------
    list of words: list of str
        The list of stemmed words in the sentence.

    '''
    stemmer = SnowballStemmer(language='english')
    return [stemmer.stem(token) for token in line]

def lemmatization(line):
    '''
    Performs lemmatization to a sentence, which group words based on their lemma.

    Parameters
    ----------
    line: str
        The sentence to perform lemmatization

    Returns
    -------
    list of words: list of str
        The list of lemmatized words in the sentence.

    '''
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in line]

def remove_stop_words(line):
    '''
    Removes stop words in a sentence, such as is, the, are, etc.
    It retains the more relevant parts of the sentence and removes
    frequently used words.

    Parameters
    ----------
    line: str
        The sentence to remove stop words.

    Returns
    -------
    list of words: list of str
        The list of words after stop words are removed.

    '''
    return [remove_stopwords(token) for token in line]

def pre_processing(line):
    '''
    Conducts the full preprocessing suite to a sentence, consisting of
    tokenization, removing stop words, lemmatization and stemming.

    Parameters
    ----------
    line: str
        The sentence to be preprocessed.

    Returns
    -------
    string: str
        The sentence after preprocessing, chained together.

    '''
    tokenizer = TreebankWordTokenizer()

    tokenized_line = tokenizer.tokenize(clean(line))
    preprocessed_line = stemming(lemmatization(remove_stop_words(tokenized_line)))
        
    return ' '.join([token for token in preprocessed_line if token != ''])

    
def keyword_filter(df, keywords, column='sentence'):
    '''
    Performs keyword filtering to each sentence in a dataframe. For each sentence, it will keep the sentence if it contains the keyword,
    and iterates through all the keywords provided.
    Once filtering is completed on the dataframe, it will then aggregate the keyword of each sentence to return a list of keywords
    that are found in the sentence.

    Parameters
    ----------
    df: pandas DataFrame
        The dataframe to perform keyword filtering on.
    keywords: list of str.
        The list of keywords that is used to filter if a sentence is relevant
        to the keyword.
    column: str
        The column in the dataframe to perform keyword filtering on.
        By default, the 'sentence' column is used.

    Returns
    -------
    filtered_df: pandas DataFrame
        The filtered dataframe containing sentences that match the keyword,
        as well as the list of keywords related to the sentence.
    '''
    filtered = []
    for s in np.array(df[column]):
        sentence = s.lower()
        for k in keywords:
            if k in sentence:
                filtered.append([s, k])
        
    filtered_df = pd.DataFrame(filtered, columns=['sentence', 'keyword(s)']).groupby(['sentence']).agg({'keyword(s)': lambda x: list(x.unique())}).reset_index()
    return filtered_df
    
def word_embedding(df, embed_column, attribute_no, embedding_model='tfidf'):
    '''
    Applies word_embedding models onto a dataframe column. Depending on the attribute number that is given, it will apply the
    corresponding saved tfidf model.

    Parameters
    -----------
    df: pandas DataFrame
        The dataframe to be used for word embedding.

    embed_column: str
        The column from the dataframe to be used for word embedding.

    attribute_no: int
        The tfidf model associated to use. The tfidf model that is selected is
        determined from the attribute_no.

    embedding_model: str
        The embedding_model to be applied to the dataframe. By default, as only tfidf is
        supported, only tfidf is accepted. Implemented for flexibility with other
        word embedding models in the future.

    Returns
    -------
    X_encoded: pandas DataFrame
        The dataframe containing word_embedding.
    '''
    ## If embedding model used is tfidf model, corresponding tfidf model will be loaded.
    if embedding_model == 'tfidf': 
        X = df[embed_column]
        X = X.apply(lambda x: x.lower())
        if attribute_no == 14:
            tfidf = pickle.load(open('pipeline/models/tfidf_14_model.sav', 'rb'))
        elif (attribute_no == 7) or (attribute_no == 15):
            tfidf = pickle.load(open('pipeline/models/tfidf_15_model.sav', 'rb'))
        elif attribute_no == 17:
            tfidf = pickle.load(open('pipeline/models/tfidf_17_model.sav', 'rb'))
        else: ## If attribute number is wrong, it means that there is no corresponding trained tfidf model.
            raise Exception(f"Wrong Model used for attribute: {attribute_no}")
        x = tfidf.transform(X)
        X_encoded = pd.DataFrame(x.toarray())
        return X_encoded
    else: ##Used for further expansion of other word embedding models.
        raise Exception("No model found")
        
def qa_filtering(df, name='deepset/roberta-base-squad2'):
    '''
    BERTQA model used to generate answers to find which company had audited the carbon emissions
    of various companies.

    Parameters
    ----------
    df: pandas DataFrame
        The dataframe to be used for BERTQA to answer who had audited the carbon emissions.

    name: str
        The name of the bert_model to be used. By default, it is assumed that the bert_model
        has been locally installed, and the bert_model used will be placed in the folder
        'pipeline/bert_model'

    Returns
    -------
    res: pandas DataFrame
        The resultant dataframe containing the answers to each sentence, to the best
        of BERTQA's ability.
    
    '''
    model_name = name
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

    res = []
    ## Setting up questions to find the company.
    q1 = 'Who audited the targets?'
    q2 = 'Who assured the targets?'
    q3 = 'Who verified the targets?'

    ## Getting answer and score for each sentence.
    for i in df['sentence']:
        QA_1 = {
            'question': q1,
            'context': i
        }
        QA_2 = {
            'question': q2,
            'context': i
        }
        QA_3 = {
            'question': q3,
            'context': i
        }

        ans1 = nlp(QA_1)['answer']
        score1 = nlp(QA_1)['score']
        ans2 = nlp(QA_2)['answer']
        score2 = nlp(QA_2)['score']
        ans3 = nlp(QA_3)['answer']
        score3 = nlp(QA_3)['score']

        ## Checking which answer has the highest score, and return the answer with the highest score.
        maxi = max([score1, score2, score3])
        if maxi == score1:
            res.append(ans1)
        elif maxi == score2:
            res.append(ans2)
        else:
            res.append(ans3)
    return res

def is_quantitative(x):
    '''
    Sentence preprocessing function to find numerical targets for companies working
    with their suppliers.

    Parameters
    ----------
    x: str
        The sentence used to find numerical targets.

    Returns
    --------
    x: str
        The list of sentences fulfilling the criteria of a set target for suppliers.
    '''
    x = x.lower()

    x = re.sub("[2][0][0-5][0-9]", "", x) #remove years
    x = re.sub("fy[0-9]+", "", x) #remove numbers that represent financial year e.g. FY21
    x = re.sub("tier\s*[0-9]", "", x) #remove numbers related to tiers
    x = re.sub("scope\s*[0-9]", "", x) #remove numbers related to scope
    x = re.sub("co2", "", x) #remove 'CO2'
    x = re.sub("cid.+", "", x) #remove 'cid'
    x = re.sub("[0-9]+[:)]|[#]+[0-9]", "", x) #remove numbers for indexing e.g. 1) or #1 or 1:

    return re.search("supplier", x) and len(re.findall(r'\d+', x)) > 0

def stop_words_removal(sentence, stop_words):
    '''
    Preprocessing function for each sentence to remove stop_words.

    Parameters
    ----------
    sentence: str
        The sentence to remove stop_words from.
    stop_words: list of str
        The list of stop_words to be removed from the sentence.

    Returns
    -------
    sentence: str
        The sentence with the stop_words removed.
    '''
    words = sentence.split()
    removed_sentence=[]
    for r in words:
        if not r in stop_words:
            removed_sentence.append(r)
            removed_sentence.append(" ")
    return "".join(removed_sentence)

def porter_stemmer(sentence):
    '''
    Preprocessing function to apply stemming using PorterStemmer.

    Parameters
    ----------
    sentence: str
        The sentence to apply stemming to.

    Returns
    -------
    sentence: str
        The sentence after stemming.
    '''
    porter = PorterStemmer()
    token_words=word_tokenize(sentence)
    token_words
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)

def custom_standardization(input_data):
    '''
    Preprocessing function to remove punctuation in sentences.

    Parameters
    ----------
    input_data: str
        The sentence to remove punctuation from.

    Returns
    -------
    sentence: str
        The sentence after removing punctuation from the sentence.
    '''
    lowercase = tf.strings.lower(input_data)
    return tf.strings.regex_replace(lowercase,
                                  '[%s]' % re.escape(string.punctuation), '')
