# PDF text extraction
from pdfminer3.layout import LAParams, LTTextBox
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager
from pdfminer3.pdfinterp import PDFPageInterpreter
from pdfminer3.converter import PDFPageAggregator
from pdfminer3.converter import TextConverter

##Image Extraction
import numpy as np
import pandas as pd
import PyPDF2
import tabula
from tabula import read_pdf
import io
from functools import reduce
from pdfminer.high_level import extract_text

import re
import fitz

class ImageExtractor:
    def __init__(self, pdf):
        self.pdf = pdf

    def run(self, output_path, keywords=[r'scope \d', 'location-based', 'market-based'], table_only=True):
        #reading pdf file to filter keywords
        pdfReader = PyPDF2.PdfFileReader(self.pdf)
        totpages = pdfReader.numPages
        
        print("Starting with file: " + self.pdf)
        page_with_keywords = []
        for p in range(pdfReader.numPages):
            text = pdfReader.pages[p].extract_text().lower()
            if any(re.search(x, text) for x in keywords):
                if (p+1) not in page_with_keywords:
                    page_with_keywords.append(p + 1)
        
        ## Filter for only tables.
        if table_only:
            table_pages = []
            for i in page_with_keywords:
                pdf = read_pdf(self.pdf, pages=i, stream=True, pandas_options={'header':'None'}, multiple_tables=True)
                if len(pdf) > 0:
                    table_pages.append(i)
            page_with_keywords = table_pages
        
        ##Extract images
        doc = fitz.open(self.pdf)
        for i in page_with_keywords:
            print(f"Extracting page: {i}")
            page = doc.load_page(i-1)
            pix = page.get_pixmap()
            output_name = self.pdf.split('/')[-1]
            output_name = output_name.split('.')[0]
            output = output_name + '_page_' + str(i) + '.png'
            pix.save(output_path + '/' + output)
            #pdf2image.convert_from_path(self.pdf, output_folder = output_path, fmt='png', first_page = i, last_page = i, output_file = str(self.pdf).split('.')[0] + str(i))
        
        print('Finished with file: ' + self.pdf)
        return ""
