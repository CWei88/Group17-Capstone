######################################### IMPORTING PACAKGES #############################
# Basic ML Packages
import os
import json

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
import io

import en_core_web_sm



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

def extract_pages_sentences(text, nlp):    
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

    pages = text.split('##PAGE_BREAK##')
    #print('Number of Pages: {}'.format(len(pages)))

    lines = []
    for i in range(len(pages)):
        page_number = i + 1
        page = pages[i]
        print(f"Extracting page number: {i}")
        
        # remove non ASCII characters
        text = remove_non_ascii(page)
        
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
    all_sentences = []

    for line in lines[:-1]: # looping through each page
        
        line = preprocess_lines(line)       
        pages_content.append(str(line).strip())

        sentences = []
        # split paragraphs into well defined sentences using spacy
        for part in list(nlp(line).sents):
            sentences.append(str(part).strip())
            
        # Only interested in full sentences and sentences with 10 to 100 words. --> filter out first page/content page
        sentences = [s for s in sentences if re.match('^[A-Z][^?!.]*[?.!]$', s) is not None]
        sentences = [s.replace('\n', ' ') for s in sentences]
        
        pages_sentences.append(sentences)
        all_sentences.extend(sentences)
    return pages_sentences, all_sentences #list, list of list where page is index of outer list, list of sentences
