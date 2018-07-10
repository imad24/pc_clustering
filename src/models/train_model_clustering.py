import click
import pandas as pd
import math
import numpy as np
import re
import itertools
from datetime import datetime,date

from sklearn.model_selection import train_test_split  

from external import kMedoids
from scipy.spatial.distance import pdist,squareform

from data import preprocessing as prp
import settings
import logging

from clusteringModel import ClusteringModel


@click.command()
@click.argument('season',type=str)
@click.option('--version',type=int)
@click.option('--k',type=int)
def main(season,version = 99, k = None):
    """ Contains all the functions of data preprocessing
    """

    #Set up file names
    s = season
    z_file_name ="p2_z_clean_%s"%s


    #Load files
    product_df = prp.load_file(z_file_name, type_="P",version = 1)


    #Get Data
    # X_train, X_test = train_test_split(product_df, test_size=0.25)
    # prp.save_file(X_test,"test",type_="P")
    X_train = product_df.copy()
    X_z = X_train.values[:,settings.n_row_headers:].astype(np.float64)

    model = ClusteringModel("kMedoids",k)

    if k is None:
        sse, silhouette = model.grid_search(X_z,k_values=np.linspace(5,15,10).astype(int))
        print(sse,silhouette)
        k = ClusteringModel.select_best_k(sse,silhouette)
        #select best k:
        print(k)
    else:
        model.fit(X_z,k=k)
        labels, _ = model.labels, model.centroids
        cluster_df = labels_to_df(labels,product_df,settings.row_headers)
        save_model(cluster_df,"p2_clusters_%s"%s,v=version)
        print("Model successfully saved")




def save_model(df,filename,v=1):
    """Saves the clustering model
    
    Arguments:
        df {Dataframe} -- Pandas dataframe containing item key, cluster and centroid columns
        filename {str} -- the file name
    
    Keyword Arguments:
        v {int} -- version of the file (default: {1})
    """

    prp.save_file(df,filename,type_="M",version = v )



def labels_to_df(labels,data_df,headers):
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



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # pylint: disable=no-value-for-parameter
    main()