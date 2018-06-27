import os
import sys
# add the 'src' directory as one where we can import modules
root_dir = os.path.join(os.getcwd(),os.pardir,os.pardir)
src_dir = os.path.join(os.getcwd(), os.pardir,os.pardir, 'src')
if src_dir not in sys.path: sys.path.append(src_dir)

import click




import pandas as pd
import math
import numpy as np

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import copy as cp

import seaborn as sns

import statsmodels.api as sm
from scipy.stats import chisquare

from sklearn.metrics import classification_report, confusion_matrix,mean_squared_error as MSE
from sklearn.metrics import precision_recall_fscore_support as report

import re
import itertools
from datetime import datetime,date


from data import preprocessing as prp

from dotenv import find_dotenv, load_dotenv
#Load env vars
load_dotenv(find_dotenv())


subfolder = os.getenv("SUBFOLDER")
PREFIX = os.getenv("PREFIX")
raw_path = os.path.join(root_dir,"data\\raw\\",subfolder)
interim_path = os.path.join(root_dir,"data\\interim\\",subfolder) 
processed_path = os.path.join(root_dir,"data\\processed\\",subfolder) 

reports_path = os.path.join(root_dir,"reports\\",subfolder)
models_path = os.path.join(root_dir,"models\\",subfolder)

@click.command()
def main():
    """ Contains all the functions of data preprocessing
    """


    #Clustering reults
    file_name = "7C_euc_p2_clustering_clean_week_v3.csv"
    df_prd_cluster = pd.read_csv(models_path+file_name, sep=';', encoding='utf-8').drop('Unnamed: 0',axis=1).set_index('Product')


    #product description
    file_name = "product_7cerf.txt"
    df_produit = pd.read_csv(raw_path+file_name, sep='\t',encoding="utf8")
    df_produit = df_produit.drop_duplicates(["Key_lvl2","Description"])


    #Join with clusters
    unbalanced = ["Description","Key_lvl7","Product Status"]#,"Sales Season"
    product_cluster = df_produit.join(df_prd_cluster,on='Key_lvl2',how='inner').reset_index(drop = True).dropna(axis = 1)
    product_cluster.drop(unbalanced, axis = 1 , inplace=True)

    #translate features
    product_cluster = prp.translate_df(product_cluster,columns=["Key_lvl3","Key_lvl4","Key_lvl5","Key_lvl6"])


    #feature engineering
    dataframe = product_cluster.set_index(["Key_lvl2"])[["Key_lvl3","Color","Size","Launch Date","Age Group","Cluster","Sales Season"]].copy()
    dataframe.index.names = ["Product"]
    features_list = ["Color","Size","Launch Date","Age Group","Person","Pname","Ptype","Price","Cluster"]
    raw_df = dataframe.copy()

    df = extract_features(raw_df)
    df = df[features_list]

    save_file(df,"clf_data",type_="P",index = True)
    print("Data set succefully made !")





def extract_features(rdf):
    data_frame = rdf.copy()
    season = rdf["Sales Season"].min()
    data_frame.drop("Sales Season",axis=1,inplace = True)
    data_frame["Person"] = data_frame["Key_lvl3"].map(lambda x: GetInfo(x,0))
    data_frame["Pname"] = data_frame["Key_lvl3"].map(lambda x: GetInfo(x,1))
    data_frame["Ptype"] = data_frame["Key_lvl3"].map(lambda x: GetInfo(x,2))
    data_frame["Price"] = data_frame["Key_lvl3"].map(lambda x: GetInfo(x,3))
    # data_frame["Launch Date"] = data_frame["Launch Date"].map(lambda x: date_to_week(x,season)).astype(str)
    data_frame["Launch Date"] = data_frame["Launch Date"].map(lambda x: get_week_number(x)).astype(str)
    data_frame.drop(["Key_lvl3"],axis=1,inplace  = True)
    
    #missing values
    data_frame.Person.fillna("Female")

    data_frame.Pname.fillna("One-Piece Pants Inside")
    
    return data_frame
    
def _first_week_of_season(season,year):
    return {
        "Autumn":date(year,9,21).isocalendar()[1],
        "Winter":date(year,12,21).isocalendar()[1],
        "Spring":date(year,3,21).isocalendar()[1],
        "Summer":date(year,6,21).isocalendar()[1]
    }[season]

def date_to_week(d,season):
    try:
        the_date = datetime.strptime(d,"%d/%m/%Y")
        first_week = _first_week_of_season(season,the_date.year)
        if(d=="01/01/1900"): return 1
        week_number = the_date.isocalendar()[1]
        return (week_number - first_week)+1
    except Exception as err:
        print(err)
        return d

def get_week_number(d):
    the_date = datetime.strptime(d,"%d/%m/%Y")
    return the_date.isocalendar()[1]

def GetInfo(key3,order,sep = " -"):
    try:
        splits = key3.split(sep)
        if len(splits)<4:
            if order == 3: res = _get_price(key3).strip()
            if order == 2: res = "Thin" 
            if order == 0: res="Female" 
            if order == 1: res = "One-piece pants inside"
        else:
            if order == 3: res =  _get_price(key3).strip()
            else: res =  splits[order].strip()
        
        return str(_redefine_group(res)).title()
    except Exception:
        print("An error occured (%d,%s)"%(order,key3))
        return None


def _get_price(s,i=0):
    try:
        regex = r"^[^\d\$]*(\$?\s?\d{1,3}\.?\d{0,2}\D{0,5}$)"
        matches  = re.findall(regex,s)
        price = matches[0].replace(" ","").upper().replace("RMB","YUAN").strip()
        return price
    except Exception as ex:
        raise ex
    
def _redefine_group(key):
    key = key.title()
    dico = {
        "Boy":"Boys",
        "Pregnant Woman" : "Pregnant",
        "Pregnant Women"  : "Pregnant",
        "Women" : "Female"
    }
    return dico[key] if key in dico else key



def save_file(data,filename,type_="I",version = 1,index=False):
    """save a dataframe into a .csv file
    
    Arguments:
        data {Dataframe} -- a Pandas dataframe
        filename {str} -- the file name
    
    Keyword Arguments:
        type_ {str} -- The data folder: (I)nterim, (P)rocessed, (R):Raw or (M)odel (default: {"I"})
        version {int} -- the file version (default: {1})
        index {bool} -- either to include the index or not (default: {False})
    """

    folder  = {
        "R" : raw_path,
        "I" : interim_path,
        "P" : processed_path,
        "M" : models_path
    }.get(type_,interim_path)

    fullname = "%s_%s_v%d.csv"%(PREFIX,filename,version)
    data.to_csv(folder+fullname, sep=";", encoding = "utf-8",index = index)


def load_file(filename,type_="I",version=1,sep=";", ext="csv",index =None):
    """Loads a csv or txt file into a dataframe
    
    Arguments:
        filename {string} -- the filename to load
    
    Keyword Arguments:
        type_ {str} -- The data folder: (I)nterim, (P)rocessed, (R):Raw or (M)odel (default: {"I"})
        version {int} -- The file version specified when saved (default: {1})
        sep {str} -- the separator in the file (default: {";"})
        ext {str} -- the extension of the file (default: {"csv"})
        Index {list} -- the columns to set as index to the dataframe
    
    Returns:
        Dataframe -- returns a pandas dataframe
    """

    folder  = {
        "R" : raw_path,
        "I" : interim_path,
        "P" : processed_path,
        "M" : models_path
    }.get(type_,interim_path)
    fullname = "%s_%s_v%d.%s"%(PREFIX,filename,version,ext)
    df = pd.read_csv(folder+fullname,sep=";",encoding="utf-8")
    if index is not None: df.set_index(index,inplace=True)

    return df



    
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    # logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()