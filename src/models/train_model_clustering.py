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
def main(season,version, k):
    """ Contains all the functions of data preprocessing
    """
    try:
        logger = logging.getLogger(__name__)
        logger.info('Running clustering model for <<%s>> version = %d...'%(season,version))

        #Load files
        clean_df = load_file("p2_clean",type_="P",version = 1).set_index(row_headers)
        assert clean_df is not None
        #Filter by season and normalize
        sclean_df = filter_by_season(clean_df,season)
        zclean_df,_  = get_scaled_series(sclean_df) 

        #Get Data
        # X_train, X_test = train_test_split(zclean_df, test_size=0.25)
        X_train = zclean_df.copy()
        X_z = X_train.values.astype(np.float64)

        logger.info("Init clustering model")
        model = ClusteringModel("kMedoids",k,init_method = settings.options["init_method"])

        # if not provided get k using grid search
        if k is None:
            logger.info("Running Hierarchical Clustering...")
            # k = model.agg_cut_off(X_z)
            logger.info("Running Grid Search to select the best number of clusters (greedy)...")
            sse, silhouette = model.grid_search(X_z,k_values=np.linspace(6,15,9).astype(int))
            k = ClusteringModel.select_best_k(sse,silhouette,best=3)
            #select best k:
            logger.info("the number of %d clusters has been selected"%k)
        logger.info("Training clustering model with %d clusters"%k)    
        model.fit(X_z,k=k, weights = None)
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


def labels_to_df(df,labels):
    """Takes the array of samples and the assigned cluster labels  and returns a dataframe of (key,cluster,centroid)
    
    Arguments:
        df {Dataframe} -- the (n_sample,m_feature) array given to the clustering
        labels {list} -- the list of cluster of labels returned by the clustering model
    
    Returns:
        Dataframe -- a dataframe of (keys,clusters,centroids)
    """
    #dicts to store clusters and theirs centers
    cluster_medoid = {}
    label_cluster = {}
    medoids = list(set(labels))
    
    #name clusters and store medoids (1)
    for i,l in enumerate(medoids):
        medoid = df.index[l]
        cluster_medoid[i+1] = medoid
        label_cluster[l] = i+1

    #creates a dataframe where each row goes like this:
    #[key, cluster, centroid]
    rows=[]
    for i,label in enumerate(labels):
        #the cluster id assigned to this label (cluster id given in (1) )
        cluster = label_cluster[label]
        #create the row: cluster and centroid
        rows.append([cluster,cluster_medoid[cluster]])
    #create the final dataframe using the entry index
    label_df = pd.DataFrame(rows,index = df.index,columns =["Cluster","Centroid"])
    return label_df

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # pylint: disable=no-value-for-parameter
    main()
