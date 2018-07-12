import itertools
import logging
import math
import re
from datetime import date, datetime

import click
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import train_test_split

import settings
from clusteringModel import ClusteringModel
from data import preprocessing as prp
from external import kMedoids


@click.command()
@click.argument('season',type=str)
@click.option('--version',type=int)
@click.option('--k',type=int)
def main(season,version = 99, k = None):
    """ Contains all the functions of data preprocessing
    """
    logger = logging.getLogger(__name__)


    #Set up file names
    s = season
    z_file_name ="p2_z_clean_%s"%s

    logger.info('Running clustering model for <<%s>> version = %d...'%(s,version))

    #Load files
    product_df = prp.load_file(z_file_name, type_="P",version = 1)






def save_model(df,filename,v=1):
    """Saves the clustering model
    
    Arguments:
        df {Dataframe} -- Pandas dataframe containing item key, cluster and centroid columns
        filename {str} -- the file name
    
    Keyword Arguments:
        v {int} -- version of the file (default: {1})
    """

    prp.save_file(df,filename,type_="M",version = v )






if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # pylint: disable=no-value-for-parameter
    main()
