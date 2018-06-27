# -*- coding: utf-8 -*-
import os
import sys
import click
# add the 'src' directory as one where we can import modules
root_dir = os.path.join(os.getcwd(),os.pardir,os.pardir)
src_dir = os.path.join(os.getcwd(), os.pardir,os.pardir, 'src')
if src_dir not in sys.path: sys.path.append(src_dir)

from data import preprocessing as prp

import pandas as pd
import math
import numpy as np

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats
from scipy.stats import mstats,chisquare
from IPython.display import display as dp

import matplotlib.pyplot as plt
import copy as cp

import seaborn as sns

import statsmodels.api as sm

from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.metrics import precision_recall_fscore_support as report

import helpers as hlp
import import_data
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
row_headers = ["Product"]


import logging

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main():#input_filepath, output_filepath
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)


    # df = translate_df(product_cluster,columns=["Key_lvl3","Key_lvl4","Key_lvl5","Key_lvl6"])
    # train_columns = ["","","","","","","","","","","","","","","","","","","","","","","","","","","","",]

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()


def load_clustering_result(f_cluster="clusters.csv",f_products=None,f_clients=None,cluster_key="Product",produit_key="Key_lvl2",client_key=""):

    clusters = load_file(f_cluster,type_="M",index = cluster_key)
    products = load_file(f_products, sep='\t',ext="txt", type_="R")
    df = products.join(clusters,on=produit_key,how='inner').reset_index(drop = True).dropna(axis = 1)

    return df

def save_file(data,filename,type_="I",version = 1,index=False):
    """save a dataframe into a .csv file
    
    Arguments:
        data {Dataframe} -- a Pandas dataframe
        filename {str} -- the file name
    
    Keyword Arguments:
        type_ {str} -- The data folder: (I)nterim, (P)rocessed, (R):Raw or (M)odel (default: {"I"})
        version {int} -- the file version (default: {1})
        index {bool} -- either the include the index or not (default: {False})
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

def load_data(filename):

    
    n_row_headers = len(row_headers)

    df = pd.read_csv(interim_path + filename , sep = ";", encoding = 'utf-8', header = 0)

    cols = df.columns.values
    cols[:n_row_headers]  = row_headers
    df.columns =cols

    return df


def trim_series(data):
    """Trims (removes complete zeros from each side) the series along the dataset 
    
    Arguments:
        data {Pandas Dataframe} -- a dataframe with only the series values
    
    Returns:
        dataframe -- returns a trimmed dataframe 
    """

    tail = 0
    head = 0
    #drop first column if zeros
    while (data.iloc[:,0] == 0).all():
        data.drop(data.columns[0], axis=1, inplace=True) 
        head +=1
    #drop last columns if zeros
    while (data.iloc[:,-1] == 0).all():
        data.drop(data.columns[-1], axis=1, inplace=True)
        tail += 1
    return data


def range_from_origin(data,range_):
    """Shifts the timeseries values to origin ie makes the first non zero value as the first one and counts "range_" values ahead
    
    Arguments:
        data {Dataframe} -- Pandas datafrale
        range_ {int} -- number of values to take into account
    
    Raises:
        error -- prints the index of the timeseries raising the error
    
    Returns:
        Dataframe -- returns a dataframe with the shifted data
    """

    N = data.shape
    centered = np.zeros((N,range_))
    i=0
    for index,row in data.iterrows():
        try:
            values = row[row!=0].index[:range_]
            r = row[values].values
            r.resize((1,range_))
            centered[i] = r
            i+=1
        except Exception as error:
            print(index)
            raise error

    centered_df = pd.DataFrame(centered)
    print("Data shifted to origin with %d values"%range_)

    return centered_df

def remove_tails(data,t = 15):
    """remove the timeseries having at least "t" zero values 
    
    Arguments:
        data {Dataframe} -- Pandas dataframe
    
    Keyword Arguments:
        t {int} -- the threshold number of zeros to consider to remove a series (default: {15})
    
    Returns:
        Dataframe -- Cleaned timeseries
    """

    mask = (data.iloc[:,-t:]==0).all(axis=1)
    df  =  data[~mask]
    print("Series With %d trailing zeros are removed"%t)
    print("Removed: %d , Remaining: %s"%(mask.astype(int).sum(),data.shape[0]))
    return df

def remove_heads(data,t = 15):
    mask = (data.iloc[:,:t] == 0).all(axis=1)
    df  =  data[~mask]
    print("Series With more than %d zeros are removed"%t)
    print("Removed: %d , Remaining: %s"%(mask.astype(int).sum(),data.shape[0]))
    return df




def moving_average(data,window):
    """Apply a moving average with window of size "windox"
    
    Arguments:
        data {Dataframe} -- Pandas dataframe
        window {int} -- size of window to apply
    
    Returns:
        Dataframe -- the new dataframe
    """

    rolled_df = data.rolling(window=window,axis=1,center = True,win_type=None).mean()
    return rolled_df.dropna(axis = 1)


def winsore_data(data,top=0.05,bottom=0.05):
    """Applies a winsorizing on data
    Winsorizing is to set all outliers to a specified percentile of the data; for example, 
    a 90% winsorization would see all data below the 5th percentile set to 
    the 5th percentile, and data above the 95th percentile set to the 95th percentile

    
    Arguments:
        data {Dataframe} -- Pandas datagframe
    
    Keyword Arguments:
        top {float} -- upper qunatile to consider (default: {0.05})
        bottom {float} -- lower quantile to consider (default: {0.05})
    
    Returns:
        Dataframe -- Winsorized dataframe
    """

    df = data.apply(mstats.winsorize,limits = (bottom,top),axis=1)
    return df


def remove_rare(data,t = 5):
    """Remove the series with less than "t" values
    
    Arguments:
        data {Dataframe} -- Pandas dataframe
    
    Keyword Arguments:
        t {int} -- Minimum number of values to consider (default: {5})
    
    Returns:
        Dataframe -- Cleaned dataframe
    """

    mask =(data.where(data==0,other=1.).sum(axis=1)<=t)
    return data[~mask]


def get_scaled_series(data):
    """Returns a standard scaled dataframe
    
    Arguments:
        data {Dataframe} -- Pandas dataframe
    
    Returns:
        Dataframe -- Scaled Dataframe
        StandardScaler -- the standard scaler used
    """

    d = data.as_matrix().astype(float)
    std_scaler = StandardScaler(with_mean=True, with_std=True).fit(d.T)
    X_z = std_scaler.transform(data.T).T
    return X_z,std_scaler

def data_with_headers(series,data,raw_df):
    """Add headers to data (only timeseries)
    
    Arguments:
        series {Numpy array} -- Numpy 2D array containing only timeseries values
        data {[type]} -- [description]
        raw_df {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """

    headers = raw_df[row_headers[::-1]].loc[data.index]
    product_df_full = pd.DataFrame(series, columns = data.columns,index=data.index)
    for label ,column in headers.iteritems():
        product_df_full.insert(0,label,column)
    return product_df_full

def display(data,head=5):
    """Displays shape and dataframe head
    
    Arguments:
        data {Dataframe} -- Pandas dataframe
    
    Keyword Arguments:
        head {int} -- number of rows to display (default: {5})
    """

    print(data.shape)
    if head>0:
        dp(data.head(head))
    else:
        dp(data)

def translate_df(df,columns,dic_path):
    """Translates specified columns in dataframe using a numpy dictionnary
    
    Arguments:
        df {Dataframe} -- Pandas dataframe
        columns {list} -- List of columns to translate
    
    Returns:
        Dataframe -- the dataframe with ONLY translated columns
    """

    try:
        tdf = df.copy()
        dico = np.load(dic_path).item()
        tans = df[columns].applymap(lambda x:dico[x])
        for index,col in tans.iteritems():
            if index in df.columns: tdf[index] = col
        return tdf
    except Exception as ex:
        print("Error when translating: ",ex)