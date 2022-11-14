import os
import re
import io
import string

## Remove warnings
import warnings
warnings.filterwarnings('ignore')

from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage

import en_core_web_sm

def extract_pdf(file):
    '''
    Pdf extraction function to extract pdf page-by-page

    Parameters
    -----------
    file: str
        The file path to which where the pdf file is located.

    Returns
    -------
    text: str
        A condensed string of text containing all the words extracted from the pdf.
    '''
    ## Load functions for pdf processing.
    resource_manager = PDFResourceManager()
    retstr = io.StringIO()
    codec = 'utf-8'
    laparams = LAParams()

    text_converter = TextConverter(resource_manager, retstr, codec=codec, laparams=laparams)
    page_interpreter = PDFPageInterpreter(resource_manager, text_converter)

    data = []

    with open(file, 'rb') as f:
        for page in PDFPage.get_pages(f, caching=True, check_extractable=False):

            page_interpreter.process_page(page)
            data.append(retstr.getvalue())

            retstr.truncate(0)
            retstr.seek(0)

    text = '##END_OF_PAGE##'.join(data)

    return text

def preprocess(sentence):
    '''
    Preprocessing function to clean up the extracted pdf data.

    Parameters
    ----------
    sentence: str
        The sentence to be preprocessed.

    Returns
    -------
    sentence: str
        The sentence after preprocessing.
    
    '''
    ## Removes header numbers
    sentence = re.sub(r'^\s?\d+(.*)$', r'\1', sentence)
    ## Strip sentence of trailing whitespace
    sentence = sentence.strip()
    ## Link back words that have been split in-between lines.
    sentence = re.sub(r'\s?-\s?', '-', sentence)
    ## Remove space before punctuation
    sentence = re.sub(r'\s?([,:;\.])', r'\1', sentence)
    ## ESG contains a lot of figures that are not relevant to grammatical structure
    sentence = re.sub(r'\d{5,}', r' ', sentence)
    ## Remove emails from text
    sentence = re.sub(r'\S*@\S*\s?', '', sentence)
    ## Remove URLs from text
    sentence = re.sub(r'(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w\.-]*)*', r' ', sentence)
    ## Consolidate multiple spacing into one
    sentence = re.sub(r'\s+', ' ', sentence)
    ## join next line with space
    sentence = re.sub(r' \n', ' ', sentence)
    sentence = re.sub(r'.\n', '. ', sentence)
    sentence = re.sub(r'\x0c', ' ', sentence)
    
    return sentence

def extract_sentences(text, nlp_model=en_core_web_sm.load()):
    '''
    Function to extract sentences from consolidated text extracted from pdf.
    It cleans the consolidated text by removing URLs, emails and irrelevant sentences,
    groups sentences into paragraphs.

    Parameters
    ----------
    text: str
        The consolidated text to be cleaned and parsed.
    nlp_model:
        The spacy model that is used to parse sentences

    Returns
    -------
    page_sentences: list of list of str
        A list containing lists of sentences. The outer list indicates the page where the sentences come from,
        where page = index of list + 1 and the inner list contains the sentence in each page.

    all_sentences: list of str
        A list containing the full text of the cleaned consolidated text, without any page number segregation.
    '''

    pages = text.split('##END_OF_PAGE##')

    sentences = []

    for i in range(len(pages)):
        page_number = i + 1
        page = pages[i]
        print(f"Extracting page number: {i}")

        ## Removes non-ASCII words from paragraph.
        text = re.sub(r'[^\x00-\x7F]+','', page)

        prev_line = ""
        for line in text.split('\n\n'):
            ## Combines consecutive lines where text may have been broken up,
            ## provided that the next line has a space at the start, and the previous line does not
            ## end with a full stop.
            if (line.startswith(' ') or not prev_line.endswith('.')):
                prev_line = prev_line + ' ' + line
            else: ## If condition is not met, we start a new index.
                sentences.append(prev_line)
                prev_line = line

        ## Ensures that the last line is stored into the array.
        sentences.append(prev_line)
        sentences.append('##PAGE_BREAK##')

    final_sentences = ' '.join(sentences).split('##PAGE_BREAK##')

    page_sentences = []
    all_sentences = []
    
    for line in final_sentences:
        line = preprocess(line)

        words = []
        ## parses paragraph into sentences using spacy.
        for partial in list(nlp_model(line).sents):
            words.append(str(partial).strip())

        w_res = []
        for w in words:
            if re.match('^[A-Z][^?!.]*[?.!]$', w) is not None:
                ## Filters for non-sentence sentences, which are
                ## sentences that do not end with ?, ., !
                w_res.append(w)

        w_res = [x.replace('\n', ' ') for x in w_res] ## Replace new line tag with blank.

        page_sentences.append(w_res)
        all_sentences.extend(w_res)

    return page_sentences, all_sentences
            

        
        
    

