import logging
import math

import click
import numpy as np
import pandas as pd

import settings
from data.preprocessing import save_file, trim_series, range_from_origin, remove_rare, smooth_series 


@click.command()
@click.argument('version',type=int)
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
def main(version = 99):#input_filepath, output_filepath

    try:

        version=99
        """ Runs data processing scripts to turn raw data from (../raw) into
            cleaned data ready to be analyzed (saved in ../processed).
        """
        logger = logging.getLogger(__name__)
        logger.info('*** Making final data set from raw data ***')

        #load data
        logger.info('loading raw data sales file...')
        product_raw_df = load_data("HistPerProduct_p2_jour.csv")

        #remove zero series
        logger.info('remove null sales...')
        non_zeros = ~(product_raw_df.fillna(0)==0).all(axis=1)
        product_df = product_raw_df.fillna(0).loc[non_zeros].copy()

        #trim zeros
        logger.info('trim empty values...')
        product_df = trim_series(product_df)

        #shit to origin
        offset = 1
        r = 16
        logger.info('shit series to origin with %d offset and range of %d ...'%(offset,r))
        product_df = range_from_origin(product_df,range_=r,offset =offset)
        
        #Save the raw series
        product_sales_raw = product_df.iloc[:,offset:].copy()
        fname = "p2_raw"
        logger.info("==> Saving raw state data to %s"%fname)
        save_file(product_sales_raw,fname,index=True,version = version)

        #remove rare sales
        logger.info("remove rare sales...")
        product_df  = remove_rare(product_df,t=5)

        #rolling average
        w = 2
        logger.info("Smoothing the series window = %d"%w)
        product_df = smooth_series(product_df,method="median",window =w)

        #save clean
        clean_filename = "p2_clean"
        logger.info("==> Saving processed data to %s"%clean_filename) 
        save_file(product_df,clean_filename,type_="P",version = version, index=True)

        #save raw values data
        raw_values_filename ="p2_series"
        logger.info("==> Saving raw values data to %s"%raw_values_filename)
        save_file(product_sales_raw.loc[product_df.index],raw_values_filename,type_="P",version=version, index=True)

    except Exception as err:
        logger.error(err)

# def save_processed_data(df,raw_df,seasons_df,version):

#     try:
#         logger = logging.getLogger(__name__)
#         seasons = set(seasons_df["Sales Season"])
#         with_seasons = df.join(seasons_df,how="left")

#         v = version
#         s="all"
#         raw_file_name ="p2_raw_%s"%s
#         z_file_name ="p2_z_clean_%s"%s
#         logger.info("\t %s, %s"%(raw_file_name,z_file_name))

#         rdf = raw_df.loc[df.index].loc[:,:]
#         zdf = df.loc[:,:]

#         save_file(rdf,raw_file_name,version = v)
#         save_file(zdf,z_file_name,type_="P",version = v,index=True)
        
#         for s in seasons:
#             sdf = (with_seasons["Sales Season"]==s)

#             raw_file_name ="p2_raw_%s"%s
#             z_file_name ="p2_z_clean_%s"%s

#             logger.info("\t %s, %s"%(raw_file_name,z_file_name))

#             rdf = raw_df.loc[df.index].loc[sdf,:]
#             zdf = df.loc[sdf,:]
#             save_file(rdf,raw_file_name,version = v)
#             save_file(zdf,z_file_name,type_="P",version = v,index=True)
#     except Exception as ex:
#         logger.error("An error occured while saving files: %s"%ex)

# def save_cleaned_data(clean_df,seasons_df,filename,version):

#     try:
#         logger = logging.getLogger(__name__) 
#         v= version
        
#         seasons = set(seasons_df["Sales Season"])
#         with_seasons = clean_df.join(seasons_df,how="left")

#         filename_all = "%s_all"%filename
#         save_file(clean_df,filename_all,type_="I",index=True,version=v)

#         for s in seasons:
#             filename_season = "%s_%s"%(filename,s)
#             sdf = (with_seasons["Sales Season"]==s)
#             df = clean_df.loc[sdf,:]
            
#             save_file(df,filename_season,type_="I",index=True,version=v)
#     except Exception as ex:
#         logger.error("An error occured while saving files: %s"%ex)

def load_data(filename):
    df = pd.read_csv(settings.interim_path + filename , sep = ";", encoding = 'utf-8', header = 0)

    cols = df.columns.values
    cols[:settings.n_row_headers]  = settings.row_headers
    df.columns =cols

    return df.set_index(settings.row_headers)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
