import pipeline.Attribute8 as Attribute8
import pipeline.Attribute12 as Attribute12
import pipeline.Attribute14 as Attribute14
import pipeline.Attribute15 as Attribute15
import pipeline.Attribute16 as Attribute16
import pipeline.Attribute17 as Attribute17
import pipeline.Attribute23 as Attribute23
import pipeline.Attribute25 as Attribute25

import numpy as np
import pandas as pd

import en_core_web_sm
class AttrGen:
    def __init__(self, df, bert_model='pipeline/bert_model'):
        self.df = df
        self.bert_model = bert_model

    def run(self):
        self.df_7 = Attribute15.Attribute15().predict(self.df)
        self.answer8, self.df_8 = Attribute8.Attribute8().predict(self.df)
        self.answer12, self.df_12 = Attribute12.Attribute12().predict(self.df)
        self.df_14 = Attribute14.Attribute14().predict(self.df)
        self.df_15 = Attribute15.Attribute15(self.bert_model).predict(self.df)
        self.df_16 = Attribute16.Attribute16().predict(self.df)
        self.df_17 = Attribute17.Attribute17().predict(self.df)
        self.df_23 = Attribute23.Attribute23().predict(self.df)
        self.score, self.df_25 = Attribute25.Attribute25().predict(self.df)

    ## Getter functions for each attribute
    def get_df7(self):
        cols = list(self.df_7.columns)
        for i in cols:
            print(self.df_7[i])
        return self.df_7

    def get_df8(self, answer=True):
        cols = list(self.df_8.columns)
        if answer:
            print(self.answer8)
        for i in cols:
            print(self.df_8[i])

        return self.df_8

    def get_df12(self, answer=True):
        cols = list(self.df_12.columns)
        if answer:
            print(self.answer12)
        for i in cols:
            print(self.df_12[i])
        return self.df_12

    def get_df14(self):
        cols = list(self.df_14.columns)
        for i in cols:
            print(self.df_14[i])

        return self.df_14
    
    def get_df15(self):
        cols = list(self.df_15.columns)
        for i in cols:
            print(self.df_15[i])

        return self.df_15
    
    def get_df16(self):
        cols = list(self.df_16.columns)
        for i in cols:
            print(self.df_16[i])

        return self.df_16

    def get_df17(self):
        cols = list(self.df_17.columns)
        for i in cols:
            print(self.df_17[i])

        return self.df_17

    def get_df23(self):
        cols = list(self.df_23.columns)
        for i in cols:
            print(self.df_23[i])

        return self.df_23

    def get_df25(self, score=True):
        cols = list(self.df_25.columns)
        if score:
            print(self.score)
        for i in cols:
            print(self.df_25[i])

        return self.df_25
        
        
