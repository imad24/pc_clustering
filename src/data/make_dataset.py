# -*- coding: utf-8 -*-
import sys 
import os
import click
# add the 'src' directory as one where we can import modules
root_dir = os.path.join(os.getcwd(),os.pardir,os.pardir)
src_dir = os.path.join(os.getcwd(), os.pardir,os.pardir, 'src')
if src_dir not in sys.path: sys.path.append(src_dir)


from dotenv import find_dotenv, load_dotenv
 
import math
import numpy as np
import pandas as pd

from data import preprocessing as prp


load_dotenv(find_dotenv())

subfolder = os.getenv("SUBFOLDER")
PREFIX = os.getenv("PREFIX")
raw_path = os.path.join(root_dir,"data\\raw\\",subfolder)
interim_path = os.path.join(root_dir,"data\\interim\\",subfolder) 
processed_path = os.path.join(root_dir,"data\\processed\\",subfolder) 

reports_path = os.path.join(root_dir,"reports\\",subfolder)
models_path = os.path.join(root_dir,"models\\",subfolder)

row_headers = ["Product"]
n_row_headers = len(row_headers)

import logging

@click.command()
#@click.argument('version', type=int)
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
def main():#input_filepath, output_filepath
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    #load data
    product_raw_df = load_data("HistPerProduct_p2_jour.csv")
    seasons_df = get_seasonsd_df()

    #remove zero series
    non_zeros = ~(product_raw_df.fillna(0)==0).all(axis=1)
    product_df = product_raw_df.fillna(0).loc[non_zeros].copy()

    #trim zeros
    product_df = prp.trim_series(product_df)

    #shit to origin
    offset = 1
    product_df = prp.range_from_origin(product_df,range_=16,offset =offset)
    #Save the raw series
    prp.save_file(product_df.iloc[:,offset:],"product_sales_raw",index=True)

    #remove rare sales
    product_df  = prp.remove_rare(product_df,t=5)

    #rolling average
    product_df = prp.smooth_series(product_df,method="average",window =2)

    save_cleaned_data(product_df,seasons_df,filename="p2_clean")

    scaled_df,_ = prp.get_scaled_series(product_df)
    
    save_processed_data(scaled_df,product_raw_df,seasons_df)


def save_processed_data(df,raw_df,seasons_df):

    try:
        seasons = set(seasons_df["Sales Season"])
        with_seasons = df.join(seasons_df,how="left")

        v = 99
        s="all"
        raw_file_name ="p2_raw_%s"%s
        z_file_name ="p2_z_clean_%s"%s


        rdf = raw_df.loc[df.index].loc[:,:]
        zdf = df.loc[:,:]

        prp.save_file(rdf,raw_file_name,version = v)
        prp.save_file(zdf,z_file_name,type_="P",version = v,index=True)
        
        for s in seasons:
            sdf = (with_seasons["Sales Season"]==s)

            raw_file_name ="p2_raw_%s"%s
            z_file_name ="p2_z_clean_%s"%s
            
            rdf = raw_df.loc[df.index].loc[sdf,:]
            zdf = df.loc[sdf,:]
            prp.save_file(rdf,raw_file_name,version = v)
            prp.save_file(zdf,z_file_name,type_="P",version = v,index=True)
    except Exception as ex:
        print_info("An error occured while saving files: %s"%ex)


def save_cleaned_data(clean_df,seasons_df,filename):

    try:
        v= 1
        
        seasons = set(seasons_df["Sales Season"])
        with_seasons = clean_df.join(seasons_df,how="left")

        filename_all = "%s_all"%filename
        prp.save_file(clean_df,filename_all,type_="I",index=True,version=v)

        for s in seasons:
            filename_season = "%s_%s"%(filename,s)
            sdf = (with_seasons["Sales Season"]==s)
            df = clean_df.loc[sdf,:]
            
            prp.save_file(df,filename_season,type_="I",index=True,version=v)
    except Exception as ex:
        print_info("An error occured while saving files: %s"%ex)


def print_info(s):
    print(s)

def get_seasonsd_df():
    seasons_df = prp.load_file("product_season",version = 1)
    seasons_df.drop(["Key_lvl1"], axis=1, inplace=True)
    seasons_df.drop_duplicates(inplace= True)
    seasons_df.set_index("Key_lvl2",inplace = True)
    seasons_df.index.names=[row_headers]

    return seasons_df

def load_data(filename):
    df = pd.read_csv(interim_path + filename , sep = ";", encoding = 'utf-8', header = 0)

    cols = df.columns.values
    cols[:n_row_headers]  = row_headers
    df.columns =cols

    return df.set_index(row_headers)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()


