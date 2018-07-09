import numpy as np
import pandas as pd
from googletrans import Translator as Translator





class DFTranslator:
    def __init__(self,destination_language = "en"):
        """Uses Google Translate to translate a dataframe
        
        Keyword Arguments:
            destination_language {str} -- the destination lang (default: {"en"})
        """

        self.destination_language=destination_language
        self.translator = None
        self.dataframe = None
        self.columns = None
        self.dictionnary = {}
        self.words = []
        self.translations = []



    def create_dictionnary(self,df,columns):
        """Creates a dictionnary from the specified columns in dataframe
        
        Arguments:
            df {Dataframe} -- Pandas dataframe containing data to translate
            columns {list} -- The list of columns to translate
        """

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
        """Saves the created dictionnary with "create_dictionnary()" to the specified path on the dist
        
        Arguments:
            path {str} -- the path in which the dic will be saved
        """

        np.save(path+'.npy', self.dictionnary) 

    def load_dictionnary(self,path):
        """Loads a saved dictionnary
        
        Arguments:
            path {str} -- the path in which the dic will be saved
        
        Returns:
            [dict] -- the loaded dictionnary
        """

        return np.load(path+'.npy').item()

    def translate_df(self,df=None,columns=None,dictionnary=None):
        """translates the specified columns of the dataframe using the provided dictionnary or the one created previously with "create_dictionnary"
        
        Keyword Arguments:
            df {Dataframe} -- Pandas dataframe containing data to translate. If none the one passed when creating the dictionnary. (default: {None})
            columns {list} -- the columns to consider when translating. If none the one passed when creating the dictionnary. (default: {None})
            dictionnary {type} -- the dictionnary to use for translation. If none the one passed when creating the dictionnary.  (default: {None})
        
        Returns:
            Dataframe -- the dataframe with the columns being translated
        """

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
            raise(ex)

