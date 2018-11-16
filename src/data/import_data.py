import logging
import math
from datetime import datetime
import os
import click
import numpy as np
import pandas as pd
import path
import settings
from data.preprocessing import save_file


@click.command()
def main():
    pass


def check_for_data():
    expected_files = ["products.csv","clients.csv","nodes.csv","series.csv","products.yaml","clients.yaml","nodes.yaml","series.yaml"]

    all_good = True
    for expected_file in expected_files:
        file_path = os.path.join(settings.raw_path,expected_file)
        all_good = all_good and os.path.isfile(file_path)
    return all_good


def import_data():   
    """ Contains all the functions of data importing
    """
    try:

        logger = settings.get_logger(__name__)

        if not (check_for_data()):
            raise Exception("The following files were not all found: %s"%("files")) 

        logger.info("*** Import data from raw files ***")
        #load raw file

        logger.info("Load raw data file (Huge file, please be patient)...")
        p1c1File = "histo_7cerf_p1c1.txt"
        df_histo_p2c1_jour = pd.read_csv(settings.raw_path + p1c1File, sep = ",", encoding = 'utf-8', header = None,dtype={0:str,2:str,3:str}).fillna(0)

        #prepare sales dataframe
        logger.info("Droping uneccessary columns...")
        sales_df=  df_histo_p2c1_jour.drop([1,3,4,5,6],axis=1)

        #set headers
        logger.info("Setting headers info...")
        end_date = "01-14-2019"
        columns = ["Product","Client"]
        nb_days = len(sales_df.columns) - len(columns)
        date_range = pd.date_range(end = end_date,periods = nb_days, freq='1w').strftime("%d/%m/%Y")
        columns.extend(date_range)
        sales_df.columns = columns

        #drop Client 0
        sales_df = sales_df[sales_df["Client"]!=0]

        #Get p1c1 keys
        p1c1 = sales_df[["Product","Client"]].dropna().drop_duplicates().astype(str).copy()

        #Product table
        logger.info("Loading products descriptions...")
        product_df = get_product_df("product_7cerf.txt")
        #save product season mapping
        save_product_season(product_df)

        #Get keys table from previous files
        p1c1p2 = p1c1.join(product_df[["Key_lvl2"]],on =["Product"]).dropna().set_index(["Product"]).astype(str)


        #save sales history
        save_p2_sales(sales_df,p1c1p2)
    
        #Get client talbe
        logger.info("Loading clients descriptions...")
        client_df = get_clients_df("client_7cerf.txt",columns =["Store Level","Business Area"] )
        cli_features = p1c1p2.join(client_df,on="Client",how="left").drop(["Client"],axis=1)

       

        #Calculate store counts
        logger.info("Saving store counts file...")
        save_storecounts(cli_features,p1c1p2)

       

        #Client counts by p2
        logger.info("Saving clients count by product...")
        save_clients_count(p1c1p2)

    except Exception as err:
        logger.error(err)




def save_p2_sales(sales,keys):
    p1_sales = sales.groupby(["Product"]).sum().fillna(0)
    df = keys[["Key_lvl2"]].join(p1_sales, how="inner").drop_duplicates()
    HistPerProduct_p2_jour = df.reset_index().drop(["Product"],axis = 1).groupby(["Key_lvl2"]).sum().fillna(0)
    filename = "HistPerProduct_p2_jour"
    save_file(HistPerProduct_p2_jour,filename,index=True)


def save_storecounts(cli_features,keys):
    ctab = pd.crosstab(cli_features.index,cli_features["Store Level"])
    ctab["Missing"] = 0    
    k1k2 = keys.drop("Client",1).drop_duplicates()
    store_counts = k1k2.astype(str).join(ctab,how="inner").fillna(0).groupby(["Key_lvl2"]).sum()
    store_counts.index.names = ['Product']
    save_file(store_counts,"store_counts",index= True)


def save_product_season(product_df):
    filename = "product_season"
    df = product_df[["Key_lvl2","Sales Season"]].drop_duplicates()
    save_file(df,filename,index=True)



def save_clients_count(keys):
    try: 
        logger = logging.getLogger(__name__)
        client_count = keys.groupby(["Key_lvl2"]).Client.nunique().to_frame()
        client_count.index.names = ["Product"]
        save_file(client_count,"p2c1_count",index = True)
    except Exception as err:
        logger.error(err)

def get_clients_df(filename,columns):
    client_df = pd.read_csv(settings.raw_path+filename, sep='\t', encoding='utf-8').set_index("Key_lvl1")
    return client_df[columns]

def get_product_df(filename):
    #product description
    df_produit = pd.read_csv(settings.raw_path+filename, sep='\t',encoding="utf8").astype(str)
    df_produit = df_produit.drop_duplicates(["Key_lvl1","Description"]).set_index(["Key_lvl1"]).astype(str)
    return df_produit

def create_product_season_file(product_df, filename="product_season"):
    df = product_df[["Key_lvl1","Key_lvl2","Sales Season"]].drop_duplicates()
    save_file(df,filename)



if __name__ == '__main__':
    main()
