import logging
import os

import click
import numpy as np
import pandas as pd


import settings
from data.preprocessing import load_file,save_file,filter_by_season,get_scaled_series
from features import tools

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split  
from sklearn.externals import joblib

import itertools

from models.classifierModel import classifierModel

@click.command()
@click.argument('season',type=str)
@click.option('--version',type=int)
def main(season,version):
    """ Train a classifier to predict a cluster
    """
    logger = settings.get_logger(__name__)
    logger.info("*** Train the classifier to predict clusters ***")

    try:
        version = 99

        s = season
        clustering_model = "p2_clusters_%s"%(s)

        #Load files
        logger.info("Loading data file...")
        #clustering result
        cluster_df = load_file(clustering_model,index=row_headers,type_="M",version = version)

        assert cluster_df is not None

        #cleaned features
        features_df = load_file('clf_features',type_="P",index = "Product")
        assert features_df is not None

        features_df.Ldate = features_df.Ldate.apply(lambda x:str(x))
        features_list = list(features_df.columns) + ["Cluster"]
        df = features_df.join(cluster_df,how="inner")[features_list]

        #get the list of features

        logger.info("Preparing data...")
        _ , numeric, categorical = tools.get_features_by_type(features_df)


        data = df[features_list].copy()
        data["Sales Season"] = s

        #prepare data
        logger.info("Encoding data...")
        X = tools.encode(data.drop(["Cluster"],axis=1),non_categorical = numeric)
        y = data["Cluster"]

        logger.info("Training classifier...")
        classifier = RandomForestClassifier(n_estimators=80,max_depth=18,min_samples_split=2, min_samples_leaf=1, criterion='gini', bootstrap=True)
        classifier.fit(X,y)


        le_encoder, ohe_encoder, scaler = tools.get_encoders("prd_le","prd_ohe","prd_scaler")
        model = classifierModel(le_encoder=le_encoder, ohe_encoder=ohe_encoder, scaler=scaler, model= classifier, non_categorical=numeric, categorical=categorical)

        filename = "classifier_%s_v%d.pkl"%(s,version)
        logger.info("Saving classifier model to << %s >>..."%filename)
        path = os.path.join(settings.models_path,filename)
        joblib.dump(model,path)


    except Exception as err:
        logger(err)


if __name__ == '__main__':
    row_headers = settings.row_headers
    # pylint: disable=no-value-for-parameter
    main()

