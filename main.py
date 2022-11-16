import numpy as np
import pandas as pd
from pathlib import Path

import pipeline.attribute_generator as ag
import pipeline.image_extraction as image_extraction
from pipeline.text_extractor import extract_sentences, extract_pdf
import en_core_web_sm

'''
The main python function to convert pdf to csv files, as well as extract relevant images from these files.

To run it, you can either double click the main.py file, or enter python main.py in your terminal.

'''

if __name__ == '__main__':
    filename = input('What is the name of the file?')
    if filename.endswith('.pdf'):
        corpus = []
        page_sentence, all_sentence = extract_sentences(extract_pdf(filename), en_core_web_sm.load())
        corpus.extend(all_sentence)
        df = pd.DataFrame(corpus, columns=['sentence'])

        image_extr = input("Do you want to extract images?[Y/N]")
        image_extr = image_extr.lower()
        if (image_extr == 'yes') or (image_extr == 'y'):
            output_file = input("Where do you want to output the images?")
            img = image_extraction.ImageExtractor(filename)
            img.run(output_file, table_only=True)
    
    elif filename.endswith('.csv'):
        df = pd.read_csv(filename)
    else:
        raise FileNotFoundError("File format not supported")

    torf = input("Have you installed deepset/roberta-base-squad2 into pipeline/bert_model folder?[Y/N]")
    torf = torf.lower()
    if (torf == 'y') or (torf == 'yes'):
        bert_model = 'pipeline/bert_model'
    else:
        bert_model = 'deepset/roberta-base-squad2'
    prgm = ag.AttrGen(df, bert_model)
    prgm.run()
    
    output_csv_path = input("Where do you want to output the csv files?")
    
    df_7 = prgm.get_df7()
    df_8 = prgm.get_df8()
    df_12 = prgm.get_df12()
    df_14 = prgm.get_df14()
    df_15 = prgm.get_df15()
    df_16 = prgm.get_df16()
    df_17 = prgm.get_df17()
    df_23 = prgm.get_df23()
    df_25 = prgm.get_df25()
         
    if not df_7.empty:
        df_7 = df_7[['sentence', 'auditors']]
    else:
        df_7 = pd.DataFrame({'sentence': [""], 'attribute':[7], 'auditors':[""]})
    if not df_8.empty:
        df_8 = df_8[['sentence']]
    else:
        df_8 = pd.DataFrame({'sentence': [""], 'attribute':[8], 'additional_info':[""]})
    if not df_12.empty:
        df_12 = df_12[['sentence']]
    else:
        df_12 = pd.DataFrame({'sentence': [""], 'attribute':[12], 'additional_info':[""]})
    if not df_14.empty:
        df_14 = df_14[['sentence', 'methodologies']]
    else:
        df_14 = pd.DataFrame({'sentence': [""], 'attribute':[14], 'methodologies':[""]})
    if not df_15.empty:
        df_15 = df_15[['sentence', 'auditors']]
    else:
        df_15 = pd.DataFrame({'sentence': [""], 'attribute':[15], 'auditors':[""]}) 
    if not df_16.empty:
        df_16 = df_16[['sentence']]
    else:
        df_16 = pd.DataFrame({'sentence': [""], 'attribute':[16], 'additional_info':[""]})             
    if not df_17.empty:
        df_17 = df_17[['sentence']]
    else:
        df_17 = pd.DataFrame({'sentence': [""], 'attribute':[17], 'additional_info':[""]})               
    if not df_23.empty:
        df_23 = df_23[['sentence']]
    else:
        df_23 = pd.DataFrame({'sentence': [""], 'attribute':[23], 'additional_info':[""]})               
    if not df_25.empty:
        df_25 = df_25[['sentence']]
    else:
        df_25 = pd.DataFrame({'sentence': [""], 'attribute':[25], 'additional_info':[""]})     

    company_name = filename.split('/')[-1]
    company_name = company_name.split('.')[0]
    company_name = '/' + company_name
    csv_name_7 = '_attribute_7.csv'
    csv_name_8 = '_attribute_8.csv'
    csv_name_12 = '_attribute_12.csv'
    csv_name_14 = '_attribute_14.csv'
    csv_name_15 = '_attribute_15.csv'
    csv_name_16 = '_attribute_16.csv'
    csv_name_17 = '_attribute_17.csv'
    csv_name_23 = '_attribute_23.csv'
    csv_name_25 = '_attribute_25.csv'
    
    attr_7 = '7'
    attr_8 = '8'
    attr_12 = '12'
    attr_14 = '14'
    attr_15 = '15'
    attr_16 = '16'
    attr_17 = '17'
    attr_23 = '23'
    attr_25 = '25'
    
    
    df_7.to_csv(output_csv_path + company_name + csv_name_7, index=False)
    df_8.to_csv(output_csv_path + company_name + csv_name_8, index=False)
    df_12.to_csv(output_csv_path + company_name + csv_name_12, index=False)
    df_14.to_csv(output_csv_path + company_name + csv_name_14, index=False)
    df_15.to_csv(output_csv_path + company_name + csv_name_15, index=False)
    df_16.to_csv(output_csv_path + company_name + csv_name_16, index=False)
    df_17.to_csv(output_csv_path + company_name + csv_name_17, index=False)
    df_23.to_csv(output_csv_path + company_name + csv_name_23, index=False)
    df_25.to_csv(output_csv_path + company_name + csv_name_25, index=False)
    
    df_7['auditors'] = ','.join(list(df_7['auditors'].unique()))
    df_14['methodologies'] = ','.join(list(df_14['methodologies'].unique()))
    df_15['auditors'] = ','.join(list(df_15['auditors'].unique()))
    df_7 = df_7.rename(columns={'auditors': 'additional_info'})
    df_14 = df_14.rename(columns={'methodologies': 'additional_info'})
    df_15 = df_15.rename(columns={'auditors': 'additional_info'})
    
    df_7['attribute'] = attr_7
    df_8['attribute'] = attr_8
    df_12['attribute'] = attr_12
    df_14['attribute'] = attr_14
    df_15['attribute'] = attr_15
    df_16['attribute'] = attr_16
    df_17['attribute'] = attr_17
    df_23['attribute'] = attr_23
    df_25['attribute'] = attr_25
    
    dfs = [df_7, df_8, df_12, df_14, df_15, df_16, df_17, df_23, df_25]
    resultant_df = pd.concat(dfs)
    resultant_df = resultant_df[['sentence', 'attribute', 'additional_info']]
    resultant_df = resultant_df.drop(columns=['additional_info'])
    resultant_df['attribute'] = resultant_df['attribute'].apply(lambda x: int(x))
    resultant_df = resultant_df.groupby('attribute')['sentence'].apply('\n'.join).reset_index()
    resultant_df = resultant_df[['sentence', 'attribute']]
    resultant_df = resultant_df.sort_values('attribute')
    
    csv_name_all = '_combined_attributes.csv'
    resultant_df.to_csv(output_csv_path + company_name + csv_name_all, index=False)

    
