# -*- coding: utf-8 -*-
import os
import click
import logging
from dotenv import find_dotenv, load_dotenv
import sys

    
import math
import copy as cp
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib.style
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans,Birch,AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import TSNE
from sklearn import preprocessing

from scipy.cluster import hierarchy
from scipy import stats
from scipy.stats import mstats


src_dir = os.path.join(os.getcwd(), os.pardir,os.pardir, 'src')
if src_dir not in sys.path: sys.path.append(src_dir)


import helpers as hlp
from external import kMedoids
from IPython.display import display

load_dotenv(find_dotenv())

root_dir = os.path.join(os.getcwd(), os.pardir,os.pardir)
# add the 'src' directory as one where we can import modules

logger = logging.getLogger(__name__)
logger.info('making final data set from raw data')

subfolder = os.getenv("SUBFOLDER")
PREFIX = os.getenv("PREFIX")
raw_path = os.path.join(root_dir,"data\\raw\\",subfolder)
interim_path = os.path.join(root_dir,"data\\interim\\",subfolder) 
processed_path = os.path.join(root_dir,"data\\processed\\",subfolder) 

reports_path = os.path.join(root_dir,"reports\\",subfolder)
models_path = os.path.join(root_dir,"models\\",subfolder)
row_headers = ["Product"]


@click.command()
def main():
    """ Contains all the functions of data preprocessing
    """

    

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()


def load_data(filename):
    
    n_row_headers = len(row_headers)

    product_raw_df = pd.read_csv(interim_path + filename , sep = ";", encoding = 'utf-8', header = 0)

    cols = product_raw_df.columns.values
    cols[:n_row_headers]  = row_headers
    product_raw_df.columns =cols

    return product_raw_df


def trim_series(data):
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

    return centered_df

def remove_tailed(data,t = 15):
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


def save_file(data,filename,type_="I",version = 1,index=False):
    folder  = {
        "R" : raw_path,
        "I" : interim_path,
        "P" : processed_path
    }.get(type_,interim_path)

    fullname = "%s_%s_v%d.csv"%(PREFIX,filename,version)
    data.to_csv(folder+fullname, sep=";", encoding = "utf-8",index = index)


def load_file(filename,type_="I",version=1):
    folder  = {
        "R" : raw_path,
        "I" : interim_path,
        "P" : processed_path
    }.get(type_,interim_path)
    fullname = "%s_%s_v%d.csv"%(PREFIX,filename,version)
    return pd.read_csv(folder+fullname,sep=";",encoding="utf-8")

def moving_average(data,window):
    rolled_df = data.rolling(window=window,axis=1,center = True,win_type=None).mean()
    return rolled_df.dropna(axis = 1)


def winsore_data(data,top=0.05,bottom=0.05):
    df = data.apply(mstats.winsorize,limits = (bottom,top),axis=1)
    return df


def remove_rare(data,t = 5):
    mask =(data.where(data==0,other=1.).sum(axis=1)<=t)
    return data[~mask]


def get_scaled_series(data):
    d = data.as_matrix().astype(float)
    std_scaler = StandardScaler(with_mean=True, with_std=True).fit(d.T)
    X_z = std_scaler.transform(data.T).T
    return X_z

def get_full_data(series,data,raw_df):
    headers = raw_df[row_headers[::-1]].loc[data.index]
    product_df_full = pd.DataFrame(series, columns = data.columns,index=data.index)
    for label ,column in headers.iteritems():
        product_df_full.insert(0,label,column)
    return product_df_full

def summary(data,head=5):
    print(data.shape)
    display(data.head(head))
