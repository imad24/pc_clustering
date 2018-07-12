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
from data.preprocessing import load_file,save_file,filter_by_season,get_scaled_series
from external import kMedoids

row_headers = settings.row_headers
@click.command()
@click.argument('season',type=str)
@click.option('--version',type=int)
@click.option('--k',type=int)
def main(season,version = 99, k = None):
    """ Contains all the functions of data preprocessing
    """
    try:

        logger = logging.getLogger(__name__)
        logger.info('Running clustering model for <<%s>> version = %d...'%(season,version))

        #Load files
        clean_df = load_file("p2_clean",type_="P",version = 1).set_index(row_headers)
        assert clean_df is not None
        #Filter and normalize
        sclean_df = filter_by_season(clean_df,season)
        zclean_df,_  = get_scaled_series(sclean_df) 

        #Get Data
        # X_train, X_test = train_test_split(zclean_df, test_size=0.25)
        X_train = zclean_df.copy()
        X_z = X_train.values.astype(np.float64)

        logger.info("Init clustering model")
        model = ClusteringModel("kMedoids",k)

        # if not provided get k using grid search
        if k is None:
            logger.info("Running Hierarchical Clustering...")
            k = model.agg_cut_off(X_z)
            logger.info("Automatic distance cut-off...")
            logger.info("the number of %d clusters has been selected"%k)
            # sse, silhouette = model.grid_search(X_z,k_values=np.linspace(5,15,10).astype(int))
            # print(sse,silhouette)
            # k = ClusteringModel.select_best_k(sse,silhouette)
            # #select best k:
            # print(k)
        logger.info("Training clustering model with %d clusters"%k)    
        model.fit(X_z,k=k)
        labels, _ = model.labels, model.centroids
        cluster_df = labels_to_df(X_train,labels)

        save_model(cluster_df,"p2_clusters_%s"%season,v=version)
        logger.info("Model with %d clusters successfully saved"%k)
    except Exception as err:
        logger.error(err)



def save_model(df,filename,v=99):
    """Saves the clustering model
    
    Arguments:
        df {Dataframe} -- Pandas dataframe containing item key, cluster and centroid columns
        filename {str} -- the file name
    
    Keyword Arguments:
        v {int} -- version of the file (default: {1})
    """

    save_file(df,filename,type_="M",version = v,index= True )



def labels_to_df_old(labels,data_df,headers):
    medoid_cluster_dict = dict()
    
    medoids = list(set(labels))
    for i,l in enumerate(medoids):
        medoid_cluster_dict[l] = i+1

    pd_tuples_list = list(data_df[headers].itertuples(index=False))
    headers_list = [tuple(x) for x in pd_tuples_list]
    
    rows=[]
    for i,h in enumerate(headers_list):
        m = labels[i]
        rows.append([h[0],medoid_cluster_dict[m],"%s"%headers_list[m]])

    label_df = pd.DataFrame(rows,columns = headers + ["Cluster","Centroid"])
    return label_df

def labels_to_df(df,labels):
    cluster_medoid = {}
    label_cluster = {}
    medoids = list(set(labels))
    
    for i,l in enumerate(medoids):
        medoid = df.index[l]
        cluster_medoid[i+1] = medoid
        label_cluster[l] = i+1

    
    rows=[]
    for i,label in enumerate(labels):
        cluster = label_cluster[label]
        rows.append([cluster,cluster_medoid[cluster]])
    label_df = pd.DataFrame(rows,index = df.index,columns =["Cluster","Centroid"])
    return label_df

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # pylint: disable=no-value-for-parameter
    main()
