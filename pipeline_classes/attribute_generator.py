import Attribute8
import Attribute12
import Attribute14
import Attribute15
import Attribute16
import Attribute17
import Attribute23
import Attribute25

import numpy as np
import pandas as pd

import en_core_web_sm
class AttrGen:
    def __init__(self, csv):
        self.df = pd.read_csv(csv)

    def run(self):
        self.df_7 = Attribute15.Attribute15().predict(self.df)
        self.answer8, self.df_8 = Attribute8.Attribute8().predict(self.df)
        self.answer12, self.df_12 = Attribute12.Attribute12().predict(self.df)
        self.df_14 = Attribute14.Attribute14().predict(self.df)
        self.df_15 = Attribute15.Attribute15().predict(self.df)
        self.df_16 = Attribute16.Attribute16(60).predict(self.df)
        self.df_17 = Attribute17.Attribute17().predict(self.df)
        self.df_23 = Attribute23.Attribute23().predict(self.df)
        self.df_25, relevant = Attribute25.Attribute25().predict(self.df)

    ## Getter functions for each attribute
    def get_df7(self):
        cols = list(self.df_7.columns)
        for i in cols:
            print(self.df_7[i])
        return self.df_7['sentence']

    def get_df8(self, answer=True):
        cols = list(self.df_8.columns)
        if answer:
            print(self.answer8)
        for i in cols:
            print(self.df_8[i])

        return self.df_8['sentence']

    def get_df12(self, answer=True):
        cols = list(self.df_12.columns)
        if answer:
            print(self.answer12)
        for i in cols:
            print(self.df_12[i])
        return self.df_12['sentence']

    def get_df14(self):
        cols = list(self.df_14.columns)
        for i in cols:
            print(self.df_14[i])

        return self.df_14['sentence']
    
    def get_df15(self):
        cols = list(self.df_15.columns)
        for i in cols:
            print(self.df_15[i])

        return self.df_15['sentence']
    
    def get_df16(self):
        cols = list(self.df_16.columns)
        for i in cols:
            print(self.df_16[i])

        return self.df_16['sentence']

    def get_df17(self):
        cols = list(self.df_17.columns)
        for i in cols:
            print(self.df_17[i])

        return self.df_17['sentence']

    def get_df23(self):
        cols = list(self.df_23.columns)
        for i in cols:
            print(self.df_23[i])

        return self.df_23['sentence']

    def get_df25(self):
        cols = list(self.df_25.columns)
        for i in cols:
            print(self.df_25[i])

        return self.df_25['sentence']
        
        
