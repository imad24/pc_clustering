import os
import sys
import click
# add the 'src' directory as one where we can import modules
root_dir = os.path.join(os.getcwd(),os.pardir,os.pardir)
src_dir = os.path.join(os.getcwd(), os.pardir,os.pardir, 'src')
if src_dir not in sys.path: sys.path.append(src_dir)

import pandas as pd
import math
import numpy as np

from datetime import datetime

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
row_headers = ["Product"]




@click.command()
def main():
    """ Contains all the functions of data importing
    """
    #load raw file
    p2c4File = "histo_7cerf_p1c1.txt"
    df_histo_p2c1_jour = pd.read_csv(raw_path + p2c4File, sep = ",", encoding = 'utf-8', header = None,dtype={0:str}).fillna(0)

    #prepare sales dataframe
    sales_df=  df_histo_p2c1_jour.drop([1,3,4,5,6],axis=1)
    N,M = sales_df.shape

    #set headers
    end_date = "01-14-2019"
    columns = row_headers.copy()
    nb_days = len(sales_df.columns) - len(row_headers)
    date_range = pd.date_range(end = end_date,periods = nb_days, freq='1w').strftime("%d/%m/%Y")
    columns.extend(date_range)
    sales_df.columns = columns

    #drop duplicates
    p1c1 = sales_df[["Product","Client"]].drop_duplicates().astype(str).copy()


    #product description
    file_name = "product_7cerf.txt"
    df_produit = pd.read_csv(raw_path+file_name, sep='\t',encoding="utf8").astype(str)
    df_produit = df_produit.drop_duplicates(["Key_lvl1","Description"])[["Key_lvl1","Key_lvl2"]].set_index(["Key_lvl1"]).astype(str)

    #keys table
    keys = p1c1.join(df_produit.astype(str),on="Product").astype(str)

    #client description
    # file_name = "client_7cerf.txt"
    # non_unique_features = []
    # unique_features = []

    # client_df = pd.read_csv(raw_path+file_name, sep='\t', encoding='utf-8').set_index("Key_lvl1")
    # features_df = client_df[["Store Level","Business Area"]]#.fillna("None")
    # cli_features = keys.join(features_df,on="Client",how="left").drop(["Client"],axis=1)



    # ctab = pd.crosstab(cli_features.Product,cli_features["Store Level"])
    # ctab["Missing"] = 0



def create_product_season_file(product_df, filename="product_season"):
    df = product_df[["Key_lvl1","Key_lvl2","Sales Season"]].drop_duplicates()
    save_file(df,filename)


def save_file(data,filename,type_="I",version = None,index=False):
    """save a dataframe into a .csv file
    
    Arguments:
        data {Dataframe} -- a Pandas dataframe
        filename {str} -- the file name
    
    Keyword Arguments:
        type_ {str} -- The data folder: (I)nterim, (P)rocessed, (R):Raw or (M)odel (default: {"I"})
        version {int} -- the file version (default: {1})
        index {bool} -- either the include the index or not (default: {False})
    """
    print("artifical save")
    return 0
    folder  = {
        "R" : raw_path,
        "I" : interim_path,
        "P" : processed_path,
        "M" : models_path
    }.get(type_,interim_path)

    fullname = "%s_%s_v%d.csv"%(PREFIX,filename,version) if version else "%s_%s.csv"%(PREFIX,filename)
    data.to_csv(folder+fullname, sep=";", encoding = "utf-8",index = index)


def load_file(filename,type_="I",version=None,sep=";", ext="csv",index =None):
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
    fullname = "%s_%s_v%d.%s"%(PREFIX,filename,version,ext) if version else "%s_%s.%s"%(PREFIX,filename,ext)
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