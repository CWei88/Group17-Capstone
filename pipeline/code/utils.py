######################################### IMPORTING PACAKGES #############################
from scipy import spatial
import pandas as pd
import os
import json
import numpy as np
import string
import warnings
warnings.filterwarnings("ignore")
import sys  
import os
import tensorflow as tf
import re
import io
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from dateutil.parser import parse
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
# PDF text extraction
from pdfminer3.layout import LAParams, LTTextBox
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager
from pdfminer3.pdfinterp import PDFPageInterpreter
from pdfminer3.converter import PDFPageAggregator
from pdfminer3.converter import TextConverter
# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
# Others
import requests
from pprint import pprint
from tqdm.notebook import tqdm
import io
import nltk
from nltk.stem.snowball import SnowballStemmer
import spacy
# spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm", disable=['ner'])
# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
from sklearn.feature_extraction import text
stop_words = text.ENGLISH_STOP_WORDS.union(stop_words)
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer

def extract_pdf(file_path):
    """
    Process raw PDF text to structured and processed PDF text to be worked on in Python.
    Parameters
    ----------
    file_path : Relative Location of File
    Return
    ------
    text : str
        processed PDF text if no error is throw
    """   

    try:
        resource_manager = PDFResourceManager()
        fake_file_handle = io.StringIO()
        codec = 'utf-8'
        laparams = LAParams()

        converter = TextConverter(resource_manager, fake_file_handle, codec=codec, laparams=laparams)
        page_interpreter = PDFPageInterpreter(resource_manager, converter)
        
        password = ""
        maxpages = 0
        caching = True
        pagenos = set()

        content = []

        with open(file_path, 'rb') as file:
            for page in PDFPage.get_pages(file,
                                        pagenos, 
                                        maxpages=maxpages,
                                        password=password,
                                        caching=True,
                                        check_extractable=False):
              
                page_interpreter.process_page(page)

                content.append(fake_file_handle.getvalue())

                fake_file_handle.truncate(0)
                fake_file_handle.seek(0)        

        text = '##PAGE_BREAK##'.join(content)

        # close open handles
        converter.close()
        fake_file_handle.close()
        
        return text

    except Exception as e:
        print(e)

        # close open handles
        converter.close()
        fake_file_handle.close()

        return ""
# nlp preprocessing
def preprocess_lines(line_input):
    """
    Helper Function to preprocess and clean sentences from raw PDF text 
    Parameters
    ----------
    line_input : str
        String that contains a sentence to be cleaned
    Return
    ------
    line : str
        Cleaned sentence
    ----------
    Sub: Substitute regular expression
    Split: Remove blank space from front and rear 
    """  
    # removing header number
    line = re.sub(r'^\s?\d+(.*)$', r'\1', line_input)
    # removing trailing spaces
    line = line.strip()
    # words may be split between lines, ensure we link them back together
    line = re.sub(r'\s?-\s?', '-', line)
    # remove space prior to punctuation
    line = re.sub(r'\s?([,:;\.])', r'\1', line)
    # ESG contains a lot of figures that are not relevant to grammatical structure
    line = re.sub(r'\d{5,}', r' ', line)
    # remove emails
    line = re.sub(r'\S*@\S*\s?', '', line)
    # remove mentions of URLs
    line = re.sub(r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*', r' ', line)
    # remove multiple spaces
    line = re.sub(r'\s+', ' ', line)
    # join next line with space
    line = re.sub(r' \n', ' ', line)
    line = re.sub(r'.\n', '. ', line)
    line = re.sub(r'\x0c', ' ', line)
    
    return line

def remove_non_ascii(text):
    """
    Helper Function to remove non ascii characters from text
    Printable will 
    """
    printable = set(string.printable) #Convert iterable to set
    return ''.join(filter(lambda x: x in printable, text))

def not_header(line):
    """
    Helper Function to remove headers
    Check if all the characters are in upper case
    """
    return not line.isupper()

def extract_pages_sentences(nlp, text):    
    """
    Extracting text from raw PDF text and store them by pages and senteces. Raw text is also cleand by removing junk, URLs, etc.
    Consecutive lines are also grouped into paragraphs and spacy is used to parse sentences.
    Parameters
    ----------
    nlp: spacy nlp model
        NLP model to parse sentences
    text : str
        Raw PDF text
    Return
    ------
    pages_content : list of str
        A list containing text from each page of the PDF report. Page number is the index of list + 1
    
    pages_sentences : list of list
        A list containing lists. Page number is the index of outer list + 1. Inner list contains sentences from each page
 
    """  
    MIN_WORDS_PER_PAGE = 500
    
    pages = text.split('##PAGE_BREAK##')
    #print('Number of Pages: {}'.format(len(pages)))

    lines = []
    for i in range(len(pages)):
        page_number = i + 1
        page = pages[i]
        
        # remove non ASCII characters
        text = remove_non_ascii(page)
        
        # if len(text.split(' ')) < MIN_WORDS_PER_PAGE:
        #     print(f'Skipped Page: {page_number}')
        #     continue
        
        prev = ""
        for line in text.split('\n\n'):
            # aggregate consecutive lines where text may be broken down
            # only if next line starts with a space or previous does not end with dot.
            if(line.startswith(' ') or not prev.endswith('.')):
                prev = prev + ' ' + line
            else:
                # new paragraph
                lines.append(prev)
                prev = line

        # don't forget left-over paragraph
        lines.append(prev)
        lines.append('##SAME_PAGE##')
        
    lines = '  '.join(lines).split('##SAME_PAGE##')
    
    # clean paragraphs from extra space, unwanted characters, urls, etc.
    # best effort clean up, consider a more versatile cleaner
    
    pages_content = []
    pages_sentences = []

    for line in lines[:-1]: # looping through each page
        
        line = preprocess_lines(line)       
        pages_content.append(str(line).strip())

        sentences = []
        # split paragraphs into well defined sentences using spacy
        for part in list(nlp(line).sents):
            sentences.append(str(part).strip())

        #sentences += nltk.sent_tokenize(line)
            
        # Only interested in full sentences and sentences with 10 to 100 words. --> filter out first page/content page
        sentences = [s for s in sentences if re.match('^[A-Z][^?!.]*[?.!]$', s) is not None]
        sentences = [s.replace('\n', ' ') for s in sentences]
        
        pages_sentences.append(sentences)
        
    return pages_content, pages_sentences #list, list of list where page is index of outer list

def preprocessing(report):
    """
    Lemmatize,lowercase and remove stopwords for pages of a report
    
    Parameters
    ----------
    report: list of str
        A list containing text from each page of the PDF report. Page number is the index of list + 1
    Return
    ------
    report_pages : list of str
        A list containing processed text from each page of the PDF report. Page number is the index of list + 1
    
    """  
    
    report_pages = []

    def para_to_sent(para):
        """
        Helper function to split paragraphs into well defined sentences using spacy
        """
        sentences = []
        for part in list(nlp(para).sents):
            sentences.append(str(part).strip())
        return sentences

    def remove_stopwords(texts):
        """
        Helper function to remove stopwords from sentence
        """
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        """
        Helper function to lemmatize text in sentence
        """
        texts_out = []
        doc = nlp(texts) 
        texts_out.append(" ".join([token.lemma_ for token in doc]))
        return texts_out

    def stemming(texts):
        stemmer = SnowballStemmer(language='english')
        revisions = [stemmer.stem(text) for text in texts]
        return revisions
    
    for page in report:

        sentences = para_to_sent(page.lower())

        # Do lemmatization keeping only noun, adj, vb, adv
        page_data = []
        for sentence in sentences : 
            data_lemmatized = lemmatization(sentence, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
            data_stemmed_lemmatized = stemming(sentences)
            page_data.extend(data_stemmed_lemmatized)
        page_para_stem_lemma = "".join(page_data)
        
        report_pages.append(page_para_stem_lemma)
    
    return report_pages

def lemmatization(text_list, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    # lemmatize text in sentence
    """https://spacy.io/api/annotation"""
    texts_out = []
    for texts in text_list:
        texts = texts.lower()
        texts_out.append(" ".join([token.lemma_ for token in nlp(texts)]))
    return texts_out

def keyword_filter(pages_sentences,keywords):
  """
  Filter sentences based on defined keywords

  Parameters
  ----------
  Keyworkds: list of keywords related to attributes

  Return
  ------
  filtered:pandas dataframe containing relevant sentence, page number, keywords found
  """
  filtered = []
  for idx, page in enumerate(pages_sentences):
    for sentence in page:
      for k in keywords:
        if k in sentence.lower():
          filtered.append([sentence, k, idx+1])

  filtered_df = pd.DataFrame(filtered,columns = ['sentence','keyword(s)','page'])\
                  .groupby(['sentence','page'])\
                  .agg({'keyword(s)':lambda x: list(x.unique())})\
                  .sort_values(['page'])

  return filtered_df

def removeStopWords(sentence):
    words = sentence.split()
    removed_sentence=[]
    for r in words:
        if not r in stop_words:
            removed_sentence.append(r)
            removed_sentence.append(" ")
    return "".join(removed_sentence)

from nltk.tokenize import sent_tokenize, word_tokenize
def stemSentence(sentence):
    porter = PorterStemmer()
    token_words=word_tokenize(sentence)
    token_words
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)
def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  return tf.strings.regex_replace(lowercase,
                                  '[%s]' % re.escape(string.punctuation), '')