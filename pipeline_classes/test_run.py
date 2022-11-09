import numpy as np
import pandas as pd

import attribute_generator as ag
import image_extraction
from pdf_text_extractor import extract_pages_sentences, extract_pdf
import en_core_web_sm

if __name__ == '__main__':
    filename = input('what is the name of the file?')
    if filename.endswith('.pdf'):
        corpus = []
        page_sentence, all_sentence = extract_pages_sentences(en_core_web_sm.load(), extract_pdf(filename))
        corpus.extend(all_sentence)
        df = pd.DataFrame(corpus, columns=['sentence'])
        print(df)

        image_extr = input("Do you want to extract images?")
        image_extr = image_extr.lower()
        if (image_extr == 'yes') or (image_extr == 'y'):
            output_file = input("Where do you want to output the images?")
            img = image_extraction.ImageExtractor(filename)
            img.run(output_file, table_only=True)
    
    elif filename.endswith('.csv'):
        df = pd.read_csv(filename)
    soft = ag.AttrGen(df)
    soft.run()
    print("This is attribute 14:")
    print(soft.get_df14())
    print("This is attribute 16:")
    print(soft.get_df16())
    print("This is attribute 23:")
    print(soft.get_df23())


    
