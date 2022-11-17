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
    
    '''
    Attribute Generation class that calls on other attribute function to generate dataframes.
    This file consolidates all the dataframes in one location to return the dataframe through this class.

    '''
    def __init__(self, df, bert_model='deepset/roberta-base-squad2'):
        '''
        Method to initialize the Attribute Generation class

        Parameters
        ----------
        df: pandas Dataframe
            The dataframe to be processed to answer the attributes given.

        bert_model: str
            The model used for BERTQA. If none is given, it is assumed that the bert_model has not been preinstalled
            onto the local computer, and the class will be extracted from the model.
        '''
        self.df = df
        self.bert_model = bert_model

    def run(self):
        '''
        Method to run each trained attribute model. 
        '''
        self.df_7 = Attribute15.Attribute15(self.bert_model).predict(self.df)
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
        '''
        Getter function for attribute 7 in dataframe form.

        Returns
        -------
        self.df_7: pandas Dataframe
            The dataframe generated through text classification for attribute 7:
            Have your Scope 1 - 2 & Scope 3 emissions been verified by a third party?
        '''
        return self.df_7

    def get_df8(self, answer=True):
        '''
        Getter function for attribute 8 in dataframe form.

        Parameters
        ----------
        answer: boolean
            If True, the answer to whether there are any sentences found for attribute 8
            will be returned

        Returns
        -------
        self.df_8: pandas Dataframe
            The dataframe generated through text classification for attribute 8:
            Do you have an active program to support increasing green space and promote biodiversity?
        '''
        if answer:
            print(self.answer8)
        return self.df_8

    def get_df12(self, answer=True):
        '''
        Getter function for attribute 12 in dataframe form.

        Parameters
        ----------
        answer: boolean
            If True, the answer to whether there are any sentences found for attribute 12
            will be returned

        Returns
        -------
        self.df_12: pandas Dataframe
            The dataframe generated through text classification for attribute 12:
            Do you have a long term (20 30 years) net zero target/commitment?
        '''
        if answer:
            print(self.answer12)
        return self.df_12

    def get_df14(self):
        '''
        Getter function for attribute 14 in dataframe form.

        Returns
        -------
        self.df_14: pandas Dataframe
            The dataframe generated through text classification for attribute 14:
            What scenario has been utilised, and what methodology was applied?
        '''
        return self.df_14
    
    def get_df15(self):
        '''
        Getter function for attribute 15 in dataframe form.

        Returns
        -------
        self.df_15: pandas Dataframe
            The dataframe generated through text classification for attribute 15:
            Are your emission reduction targets externally verified/assured? 
        '''

        return self.df_15
    
    def get_df16(self):
        '''
        Getter function for attribute 16 in dataframe form.

        Returns
        -------
        self.df_16: pandas Dataframe
            The dataframe generated through text classification for attribute 16:
            Do you have a low carbon transition plan? 
        '''

        return self.df_16

    def get_df17(self):
        '''
        Getter function for attribute 17 in dataframe form.

        Returns
        -------
        self.df_17: pandas Dataframe
            The dataframe generated through text classification for attribute 17:
            Do you provide incentives to your senior leadership team for the management of climate related issues? 
        '''

        return self.df_17

    def get_df23(self):
        '''
        Getter function for attribute 23 in dataframe form.

        Returns
        -------
        self.df_23: pandas Dataframe
            The dataframe generated through text classification for attribute 23:
            Does your transition plan include direct engagement with suppliers to drive them to reduce their emissions,
            or even switching to suppliers producing low carbon materials?
        '''

        return self.df_23

    def get_df25(self, score=True):
        '''
        Getter function for attribute 25 in dataframe form.

        Parameters
        ----------
        score: Boolean
            If True, the score for attribute 25 will be returned. The score returned follows a scale with the following meaning:
            1 - The company has identified climate-related issues.
            2 - The company has guidelines to monitor and regulate their value chain
            3 - The company has specific initiatives and works directly with their value chain.

        Returns
        -------
        self.df_25: pandas Dataframe
            The dataframe generated through text classification for attribute 25:
            Do you engage with your value chain on climate related issues?
        '''
        if score:
            print(self.score)

        return self.df_25
        
        
