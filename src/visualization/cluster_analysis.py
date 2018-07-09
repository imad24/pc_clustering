

import pandas as pd
import math
import numpy as np

import matplotlib.pyplot as plt
import copy as cp
import matplotlib.cm as cm
import seaborn as sns

from math import sqrt


from features import tools
import prince



def cluster_plot(df,centroid_only= False,tick_frequency = 3,top = None, Normalized = True):
    try: 
        # TODO: Create a validator class ?
        for c in ["Cluster","Centroid"]:
            if c not in df.columns:
                raise ValueError("<<%s>> column not found in dataframe"%c)

        clusters = list(set(df.Cluster))
        if top: 
            clusters = np.random.randint(1,len(clusters),size= top)
        else: 
            top = df.shape[0]
        nc = min(len(clusters),top)
        
        plt.figure(figsize=(15,nc*4))
        for i,c in enumerate(clusters):
            plt.subplot(nc,1,i+1)
            cdf = df[df.Cluster==c]
            medoid = cdf.Centroid.iloc[0]
            plt.title("Cluster %d (%s): %d items"%(c,medoid,cdf.shape[0]))
            if (centroid_only):
                row = cdf.loc[medoid]
                values = row.values[:-2]
                if (Normalized): values = values/values.std()
                plt.plot(values)
            else:    
                for _ , row in cdf.iterrows():
                    values = row.values[:-2]
                    if (Normalized): values = values/values.std()
                    plt.plot(values)
    except ValueError as ex:
        print(ex)



def plot_features_distribution(dataframe):

    df = dataframe.fillna("Na").drop("Cluster",axis=1).copy()
    _, numeric, categorical = tools.get_features_by_type(df)

    _,M = df.shape
    n_rows = int(M/4)+1
    plt.figure(figsize=(18,n_rows * 5))

    for i,f in enumerate(categorical):
        plt.subplot(n_rows,4,i+1)
        plt.title(f)
        df[f].value_counts().plot(kind="Bar")

    for f in numeric:
        i+=1
        plt.subplot(n_rows,4,i+1)
        plt.title(f)
        df[f].plot(kind="Hist")
    plt.tight_layout()


def plot_cluster_over_features(df,clusters = [], pthreshold=0.05):
    _, _, categorical = tools.get_features_by_type(df.drop("Cluster",1))

    if (len(clusters)==0): 
        clusters = list(set(df.Cluster))

    for c in clusters:
        i=0
        c_df = df[df["Cluster"]==c]
        if len(c_df.index)==0: 
            print('This cluster dosnt exist')
            continue
        for feature in categorical:

            original_dist = df[feature].value_counts().sort_index()
            dist = c_df[feature].value_counts(dropna = False).reindex(original_dist.index,fill_value = 1).sort_index()

            _, p, v_cramer = tools.get_significance(list(dist.values),list(original_dist.values))

            if p<pthreshold:
                # plt.subplot(n_rows,3,i+1)
                _,ax = plt.subplots()
                ax2 = ax.twinx()
                dist.plot(kind="Bar",ax=ax, alpha =0.5 ,title ="Cluster: %d - %s pvalue = %.2f"%(c,feature,v_cramer),width=0.5)
                original_dist.plot(kind="Bar",ax=ax2,color='green',alpha = 0.5,width = 0.4,align="edge")
                plt.grid(False)
                i+=1
            

def plot_feature_over_clusters(df,feature, pthreshold=0.05):
    for c in list(set(df.Cluster)):
        i=0
        c_df = df[df.Cluster==c]
        original_dist = df[feature].value_counts().sort_index()
        dist = c_df[feature].value_counts(dropna = False).reindex(original_dist.index,fill_value = 1).sort_index()

        _, p, v_cramer = tools.get_significance(list(dist.values),list(original_dist.values))

        if p<pthreshold:
            _,ax = plt.subplots()   
            ax2 = ax.twinx()
            dist.plot(kind="Bar",ax=ax, alpha =0.5 ,title ="Cluster: %d - %s pvalue = %.2f"%(c,feature,v_cramer))
            original_dist.plot(kind="Bar",ax=ax2,color='green',alpha = 0.5,title ="Cluster: %d - %s pvalue = %.2f"%(c,feature,p))
            plt.grid(False)
            i+=1                 


def plot_modalities(df,pthreashold = 0.3,min_dust  = True, n_min_dist = 3, min_members = 4):
    _ , _ , categorical = tools.get_features_by_type(df.drop("Cluster",1))
    for feature in categorical:
        ctab = pd.crosstab(df[feature],df.Cluster)
        for index,row in ctab.iterrows():
            dist = row.values
            _, p, v_cramer = tools.get_significance(list(dist))

            md = np.count_nonzero(dist)<=n_min_dist and np.max(dist)>min_members
            if p<pthreashold and (md and min_dust):
                plt.figure()
                ctab.loc[index].plot(kind="Bar",title="%s: %s  Distribution - pvalue = %.9f"%(feature,index,v_cramer))


def mca_plot(df):
    mca = prince.MCA(df)
    mca.plot_relationship_square()