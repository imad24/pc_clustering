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

    #remove zero series
    non_zeros = ~(product_raw_df.fillna(0).iloc[:,n_row_headers:]==0).all(axis=1)
    product_df = product_raw_df.fillna(0).iloc[:,n_row_headers:].loc[non_zeros].copy()

    #trim zeros
    product_df = prp.trim_series(product_df)

    #shit to origin
    offset = 2
    product_df = prp.range_from_origin(product_df,range_=16,offset =offset)
    #Save the raw series
    product_sales_raw = product_df.join(product_raw_df,how="inner").set_index("Product")[product_df.columns[offset:]]
    prp.save_file(product_sales_raw,"product_sales_raw",index=True)

    #remove rare sales
    product_df  = prp.remove_rare(product_df,t=5)

    #rolling average
    product_df = prp.smooth_series(product_df,method="average",window =3)

    # save_cleaned_data(product_df);


def load_data(filename):
    df = pd.read_csv(interim_path + filename , sep = ";", encoding = 'utf-8', header = 0)

    cols = df.columns.values
    cols[:n_row_headers]  = row_headers
    df.columns =cols

    return df.set_index(row_headers)


def save_cleaned_data(clean_df,raw_df):
    keys = raw_df[row_headers]
    product_clean_df = keys.join(clean_df,how="inner")

    seasons_df = prp.load_file("product_season",version = 1)
    seasons_df.drop(["Key_lvl1"], axis=1, inplace=True)
    seasons_df.drop_duplicates(inplace= True)
    seasons_df.set_index("Key_lvl2",inplace = True)

    seasons = set(seasons_df["Sales Season"])
    with_seasons = product_clean_df.join(seasons_df,how="left",on=row_headers)

    clean_full_file_name = "product_p2_clean_full_all"
    prp.save_file(product_clean_df,clean_full_file_name,type_="P")

    for s in seasons:
        clean_full_file_name = "product_p2_clean_full_%s"%s
        sdf = (with_seasons["Sales Season"]==s)
        df = product_clean_df.loc[sdf,:]
        print(s,len(df))
        prp.save_file(df,clean_full_file_name)



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()


