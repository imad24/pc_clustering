import pandas as pd
import math
import numpy as np

from sklearn import metrics

import matplotlib.pyplot as plt
import copy as cp
import matplotlib.cm as cm
import seaborn as sns

from math import sqrt

#import statsmodels.api as sm



def cluster_2d_plot(X,labels, inertia = 0,info=["","",""]):
    plt.figure(figsize=(16,7))
    colors = [str(item/255.) for item in labels]
    
    
    
    plt.suptitle("Clustering Method: (%s,%s,%s)"%(info[0],info[1],info[2]),size=14) 
    plt.subplot(1,2,1)
    plt.scatter(X[:,0],X[:,1],cmap ="Paired" ,c=colors)
    plt.xlabel("PCA01")    
    plt.ylabel("PCA02")

    plt.subplot(1,2,2)
    plt.scatter(X[:,0],X[:,2],cmap ="Paired" ,c=colors)
    plt.xlabel("PCA01")    
    plt.ylabel("PCA03") 
   

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



def decorate_plot(cols,tick_frequency = 3, rotation = 70,color='lightblue'):
    weeks = np.arange(len(cols))
    for x in weeks: 
        if (x+1)%12 == 0: plt.axvline(x,c=color,linestyle='--')
    plt.xticks(weeks[::tick_frequency], cols[::tick_frequency], rotation = rotation)



def circleOfCorrelations(components, explained_variance, cols):
    plt.figure(figsize=(10,10))
    plt.Circle((0,0),radius=10, color='k', fill=False)
    plt.axis('equal')
    circle1=plt.Circle((0,0),radius=1, color='k', fill=False)
    fig = plt.gcf()
    fig.gca().add_artist(circle1)
    plt.axhline(y=0,c='k')    
    plt.axvline(x=0,c='k')

    for idx in range(len(components)):
        x = components[idx][0]
        y = components[idx][1]
        #plt.plot([0.0,x],[0.0,y],'k-')
        plt.plot(x, y, 'rx')
        month = columnToMonth(cols[idx])
        plt.annotate("%02d"%month, xy=(x,y))
    plt.xlabel("PC-0 (%s%%)" % str(explained_variance[0])[:4].lstrip("0."))
    plt.ylabel("PC-1 (%s%%)" % str(explained_variance[1])[:4].lstrip("0."))
    plt.xlim((-1.5,1.5))
    plt.ylim((-1.5,1.5))
    plt.title("Circle of Correlations")

def columnToMonth(txt):
    weekNumber = int(txt[-2:])
    month = int((weekNumber*7)/30 + 1)
    return month

def GetMostCorrelatedTo(X_embedded,component,index,n=10,absl = True ):
    ly = X_embedded.shape[1]+1
    df_Xpca = pd.DataFrame(X_embedded,index=index,columns = np.arange(1,ly))
    return df_Xpca.abs().nlargest(n,component) if absl else df_Xpca.nlargest(n,component)





    # from matplotlib.patches import Ellipse