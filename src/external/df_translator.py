import numpy as np
import pandas as pd
from googletrans import Translator as Translator





class DFTranslator:
    def __init__(self,destination_language = "en"): 
        self.destination_language=destination_language
        self.translator = None
        self.dataframe = None
        self.columns = None
        self.dictionnary = {}
        self.words = []
        self.translations = []



    def create_dictionnary(self,df,columns):
        values = df[columns].drop_duplicates().values.ravel()
        self.words = list(np.unique(values))
        self.columns = columns
        self.translator = Translator()
        t = self.translator.translate(self.words,dest=self.destination_language)
        self.translations =[]
        for u in t:
            self.translations.append(u.text)

        for i,ts in enumerate(self.translations):
            self.dictionnary[self.words[i]] = ts 


    def save_dictionnary(self,path):
        np.save(path+'.npy', self.dictionnary) 

    def load_dictionnary(self,path):
        return np.load(path+'.npy').item()

    def translate_df(self,df=None,columns=None,dictionnary=None):
        try:
            if df is None : df=self.dataframe
            if columns is None: columns = self.columns
            if dictionnary is None: dictionnary = self.dictionnary

            tdf = df.copy()
            dico = dictionnary
            tans = df[columns].applymap(lambda x:dico[x])
            for index,col in tans.iteritems():
                if index in df.columns: tdf[index] = col
            return tdf
        except Exception as ex:
            print("Error when translating: ",ex)