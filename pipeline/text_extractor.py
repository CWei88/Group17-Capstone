import os
import re
import io
import string

from pdfminer3.converter import TextConverter
from pdfminer3.layout import LAParams
from pdfminer3.pdfdocument import PDFDocument
from pdfminer3.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfparser import PDFParser

import en_core_web_sm

def extract_pdf(file):
    resource_manager = PDFResourceManager()
    retstr = io.StringIO()
    codec = 'utf-8'
    laparams = LAParams()

    text_converter = TextConverter(resource_manager, retstr, codec=codec, laparams=laparams)
    page_interpreter = PDFPageInterpreter(resource_manager, text_converter)

    data = []

    with open(file, 'rb') as f:
        for page in PDFPage.get_pages(f, set(),
                                      caching=True, check_extractable=False):

            page_interpreter.process_page(page)
            data.append(retstr.getvalue())

            retstr.truncate(0)
            retstr.seek(0)

    text = '##END_OF_PAGE##'.join(data)

    f.close()
    text_converter.close()
    retstr.close()

    return text

def preprocess(sentence):
    # Removes header numbers
    sentence = re.sub(r'^\s?\d+(.*)$', r'\1', sentence)
    # Strip sentence of trailing whitespace
    sentence = sentence.strip()
    # Link back words that have been split in-between lines.
    sentence = re.sub(r'\s?-\s?', '-', sentence)
    # Remove space before punctuation
    sentence = re.sub(r'\s?([,:;\.])', r'\1', sentence)
    # ESG contains a lot of figures that are not relevant to grammatical structure
    sentence = re.sub(r'\d{5,}', r' ', sentence)
    # Remove emails from text
    sentence = re.sub(r'\S*@\S*\s?', '', sentence)
    # Remove URLs from text
    sentence = re.sub(r'(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w\.-]*)*', r' ', sentence)
    # Consolidate multiple spacing into one
    sentence = re.sub(r'\s+', ' ', sentence)
    # join next line with space
    sentence = re.sub(r' \n', ' ', sentence)
    sentence = re.sub(r'.\n', '. ', sentence)
    sentence = re.sub(r'\x0c', ' ', sentence)
    
    return sentence

def extract_sentences(text, nlp_model=en_core_web_sm.load()):

    pages = text.split('##END_OF_PAGE##')

    sentences = []

    for i in range(len(pages)):
        page_number = i + 1
        page = pages[i]
        print(f"Extracting page number: {i}")

        text = re.sub(r'[^\x00-\x7F]+','', page)

        prev_line = ""
        for line in text.split('\n\n'):
            if (line.startswith(' ') or not prev_line.endswith('.')):
                prev_line = prev_line + ' ' + line
            else:
                sentences.append(prev_line)
                prev = line

        sentences.append(prev_line)
        sentences.append('##PAGE_BREAK##')

    final_sentences = ' '.join(sentences).split('##PAGE_BREAK##')

    page_sentences = []
    all_sentences = []

    for line in final_sentences:
        line = preprocess(line)

        words = []
        for partial in list(nlp_model(line).sents):
            words.append(str(partial).strip())

        words = [w for w in words if re.match('^[A-Z][^?!.]*[?.!]$', w) is not None]
        w_res = [x.replace('\n', ' ') for x in words]

        page_sentences.append(w_res)
        all_sentences.extend(w_res)

    return page_sentences, all_sentences
            

        
        
    

