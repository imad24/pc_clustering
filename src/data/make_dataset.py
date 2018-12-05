import logging
import math

import click
import numpy as np
import pandas as pd

import settings
from data.preprocessing import save_file, trim_series, range_from_origin, remove_rare, smooth_series 




# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
@click.command()
@click.argument('version',type=int)
def main(version):#version
    try:
        """ Runs data processing scripts to turn raw data from (../raw) into
            cleaned data ready to be analyzed (saved in ../processed).
        """
        logger = settings.get_logger(__name__)
        logger.info("*** Making the final dataset from raw data ***")

        #load data
        logger.info('loading raw data sales file...')
        product_raw_df = load_data("7S_HistPerProduct_p2_jour.csv")

        #remove zero series
        logger.info('remove null sales...')
        non_zeros = ~(product_raw_df.fillna(0)==0).all(axis=1)
        product_df = product_raw_df.fillna(0).loc[non_zeros].copy()

        #trim zeros
        logger.info('trim empty values...')
        product_df = trim_series(product_df)

        #shit to origin
        offset = settings.options["offset"]
        r = settings.options["range"]
        logger.info('shit series to origin with %d offset and range of %d ...'%(offset,r))
        product_df = range_from_origin(product_df,range_=r,offset =offset)
        
        #Save the raw series
        product_sales_raw = product_df.iloc[:,offset:].copy()
        fname = "p2_raw"
        logger.info("==> Saving raw state data to %s"%fname)
        save_file(product_sales_raw,fname,index=True)

        #remove rare sales
        logger.info("remove rare sales...")
        product_df  = remove_rare(product_df,t=6)

        #rolling average
        w = settings.options["windows_size"]
        smoothing = settings.options["smoothing_method"]
        logger.info("Smoothing the series window = %d"%w)
        product_df = smooth_series(product_df,method=smoothing,window =w)

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


def load_data(filename):
    df = pd.read_csv(settings.interim_path + filename , sep = ";", encoding = 'utf-8', header = 0)

    cols = df.columns.values
    cols[:settings.n_row_headers]  = settings.row_headers
    df.columns =cols

    return df.set_index(settings.row_headers)

if __name__ == '__main__':
    main()
