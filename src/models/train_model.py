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

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import copy as cp

import seaborn as sns

import statsmodels.api as sm
from scipy.stats import chisquare

from sklearn.metrics import classification_report, confusion_matrix,  silhouette_score,precision_recall_fscore_support as report
from sklearn.model_selection import train_test_split  


import helpers as hlp
from data import preprocessing as prp,import_data
from dotenv import find_dotenv, load_dotenv


from clusteringModel import ClusteringModel
#Load env vars
load_dotenv(find_dotenv())

subfolder = os.getenv("SUBFOLDER")
PREFIX = os.getenv("PREFIX")
raw_path = os.path.join(root_dir,"data\\raw\\",subfolder)
interim_path = os.path.join(root_dir,"data\\interim\\",subfolder) 
processed_path = os.path.join(root_dir,"data\\processed\\",subfolder) 

reports_path = os.path.join(root_dir,"reports\\",subfolder)
models_path = os.path.join(root_dir,"models\\",subfolder)


from external import kMedoids
from scipy.spatial.distance import pdist,squareform


@click.command()
def main():
    """ Contains all the functions of data preprocessing
    """

    v=1

    #Set up file names
    season = "Summer"
    raw_file_name ="product_p2_raw_%s"%season
    clean_file_name = "product_p2_clean_%s"%season
    z_file_name ="product_z_p2_clean_%s"%season

    row_headers = ['Product']
    n_row_headers = len(row_headers)


    #Load files
    product_raw_df = prp.load_file(raw_file_name,version = v)
    product_df = prp.load_file(clean_file_name,version = v)
    product_df_full = prp.load_file(z_file_name, type_="P",version = v)

    #Get Data
    X_train, X_test = train_test_split(product_df_full, test_size=0.25)
    # prp.save_file(X_test,"test",type_="P")
    X_train = product_df_full.copy()
    X_z = X_train.values[:,n_row_headers:].astype(np.float64)


    model = ClusteringModel("kMedoids",9)



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    # logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()



def get_distances(X,distance = "euclidean"):
    return squareform(pdist(X, distance))



def _getSSE(samples,centroids):
    return np.sum( (samples-centroids)**2)

def save_model(df,filename,v=1):
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



