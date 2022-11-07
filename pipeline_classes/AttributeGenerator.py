import Attribute8
import Attribute14
import Attribute15
import Attribute16
import Attribute17
import Attribute23
import Attribute25

import numpy as np
import pandas as pd
class AttributeGenerator:

    def __init__(self, csv):
        self.df = pd.read_csv(csv)

    def run(self):
        self.df_7 = Attribute15.Attribute15().predict(df)
        self.df_8 = Attribute8.Attribute8().predict(df)
        self.df_14 = Attribute14.Attribute14().predict(df)
        self.df_15 = Attribute15.Attribute15().predict(df)
        self.df_16 = Attribute16.Attribute16().predict(df)
        self.df_17 = Attribute17.Attribute17().predict(df)
        self.df_23 = Attribute23.Attribute23().predict(df)
        self.df_25, relevant = Attribute25.Attribute25().predict(df)

    ## Getter functions for each attribute
    def get_df7(self):
        cols = list(self.df_7.columns)
        for i in cols:
            print(self.df_7[cols])
        return self.df_7['sentence']

    def get_df8(self):
        cols = list(self.df_8.columns)
        for i in cols:
            print(self.df_8[cols])

        return self.df_8['sentence']

    def get_df14(self):
        cols = list(self.df_14.columns)
        for i in cols:
            print(self.df_14[cols])

        return self.df_14['sentence']
    
    def get_df15(self):
        cols = list(self.df_15.columns)
        for i in cols:
            print(self.df_15[cols])

        return self.df_15['sentence']
    
    def get_df16(self):
        cols = list(self.df_16.columns)
        for i in cols:
            print(self.df_16[cols])

        return self.df_16['sentence']

    def get_df17(self):
        cols = list(self.df_17.columns)
        for i in cols:
            print(self.df_17[cols])

        return self.df_17['sentence']

    def get_df23(self):
        cols = list(self.df_23.columns)
        for i in cols:
            print(self.df_23[cols])

        return self.df_23['sentence']

    def get_df25(self):
        cols = list(self.df_25.columns)
        for i in cols:
            print(self.df_25[cols])

        return self.df_25['sentence']


AttributeGenerator('ubm.csv').run().get_df14()        
        
        
