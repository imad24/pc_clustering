import copy as cp
import math
from math import sqrt
import click 
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error as MSE

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors

import settings
from data.preprocessing import (filter_by_season, get_scaled_series, load_file,
                                save_file, display_df)
from features import tools

from models.predict_model_estimator import predict_std

sns.set()


range_ = settings.options["range"]
n_pred = settings.options["n_pred"]

row_headers = settings.row_headers
n_points = settings.options["n_points"]

offset = (n_pred * 2) + 2


@click.command()
@click.argument('season',type=str)
def main(season):
    return
    # clustering_model = "nb_p2_clusters_%s" % (season)

    # global raw_df
    # global series_df
    # global clean_df
    # global sclean_df
    # global zclean_df
    # global cluster_df


    # #Load files
    # raw_df = load_file("p2_raw").set_index(row_headers)
    # series_df =load_file("p2_series", type_="P", version = 1).set_index(row_headers)
    # clean_df = load_file("p2_clean", type_="P", version = 1).set_index(row_headers)

    # #Filter and normalize
    # sclean_df = filter_by_season(clean_df,season)
    # zclean_df,_ =  get_scaled_series(clean_df)
    

    # #clustering result
    # cluster_df = load_file(clustering_model,index=row_headers,type_="M",version = 1)

def error_evaluation(df):
    series_df = df.iloc[:, -range_:]

    
    guess = 1

    s_true = np.zeros((series_df.shape))
    s_pred = np.zeros((series_df.shape))
    RMSE = []
    PRMSE = []
    CORR = []
    i = 0
    for _,values in df.iterrows():
        
        centroid = values["PR%d"%guess]
        cluster = values["Centroid"]
        
        #Getting the series raw, centroid of actual cluster, centroid of predicted cluster
        series = (values[offset:]).astype(np.float64)#/values[offset:].std()).astype(np.float64)
        c_series = (series_df.loc[cluster]).astype(np.float64)#/series_df.loc[cluster].std()).astype(np.float64)
        predicted_series = (series_df.loc[centroid]).astype(np.float64)#/series_df.loc[centroid].std()).astype(np.float64)
        
        s_true[i] = series
        s_pred[i] = predicted_series

        n = len(series)
        rmse = math.sqrt(MSE(series,c_series)/n)
        prmse = math.sqrt(MSE(series,predicted_series)/n)
        corr = np.corrcoef(series,predicted_series)[0][1]
        
        
        RMSE.append(rmse)
        PRMSE.append(prmse)
        CORR.append(corr)

        i+=1
    return RMSE,PRMSE,CORR


def prediction_plot(test_df, series_df, index =[], standard = True):
    # cast bool to int 0:False and 1:True
    scale = 1 if standard else 0

    guess1 = test_df["PR1"]
    cluster = test_df["Centroid"]

    #Get the std from the built predictor
    pstd = predict_std(index)[0]
    
    #actual series
    series = series_df.loc[index] /  (series_df.loc[index].std() ** scale)
    c_series = series_df.loc[cluster] / (series_df.loc[cluster].std())

    #curve of each prediction
    p1 = series_df.loc[guess1] / (series_df.loc[guess1].std())

    if not standard:
        p1 = p1 * pstd
        c_series *= pstd
        plt.plot(p1,label="P1_pstd",ls='--',c='g')
    else:
        plt.plot(p1,label="P1",ls='--')
        plt.plot(c_series,label="Centroid",c='grey',ls='-.')
        plt.plot(series,label="Series",c='m')
        plt.legend(loc=0)



def predictions_plot(test_df, series_df, standard=True, nearest=False, all=False ):

    # if show all predictions then standardize
    if nearest:
        standard = False

    # cast bool to int 0:False and 1:True
    scale = 1 if standard else 0

    #Prepare figure
    plt.figure(figsize=(18,2 * 4))

    i=1
    for index,values in test_df.iterrows():
        #Get the curves
        guess1 = values["PR1"]
        guess2 = values["PR2"]
        guess3 = values["PR3"]
        guess4 = values["PR4"]
        cluster = values["Centroid"]
        
        #Get the std from the built predictor
        pstd = predict_std(index)[0]
        
        #actual series
        series = series_df.loc[index] /  (series_df.loc[index].std() ** scale)
        c_series = series_df.loc[cluster] / (series_df.loc[cluster].std())
        
        #curve of each prediction
        p1 = series_df.loc[guess1] / (series_df.loc[guess1].std())
        p2 = series_df.loc[guess2] / (series_df.loc[guess2].std())
        p3 = series_df.loc[guess3] / (series_df.loc[guess3].std())
        p4 = series_df.loc[guess4] / (series_df.loc[guess4].std())

        

        
        p_array = pd.DataFrame([p1,p2,p3,p4],columns = series_df.columns)
            
        # put in calculated scale
        plt.subplot(4,3,i)
        plt.title("Product: %s" % (index))

        #Adjusted
        if nearest:
            nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
            nn.fit(p_array.iloc[:,:n_points])  
            first_points = series[:n_points]/ (series[:n_points].std())
            closest = nn.kneighbors([first_points], 1, return_distance=False)[0][0] 
            cstd = (series[:n_points]/p_array.iloc[closest,:n_points]).median()
            #multiply by the caluclated std
            p1c = p1 * cstd 
            plt.plot(p1c,label="P1_cstd",ls='--',c='b')
            plt.plot(p_array.iloc[closest]*cstd,label="Closest",c='y',ls=':')

        if not standard:
            p1 = p1 * pstd
            p2 = p2 * pstd
            p3 = p3 * pstd
            p4 = p4 * pstd
            c_series *= pstd

            plt.plot(p1,label="P1_pstd",ls='--',c='g')
        else:
            plt.plot(p1,label="P1",ls='--')
        
        if all:
            plt.plot(p2,label="P2",ls='--')
            plt.plot(p3,label="P3",ls='--')
            plt.plot(p4,label="P4",ls='--')

        plt.plot(c_series,label="Centroid",c='grey',ls='-.')

        plt.plot(series,label="Series",c='m')

        plt.legend(loc=0)
        i+=1
    plt.tight_layout()

if __name__ == "main":
    # pylint: disable=no-value-for-parameter
    main()
