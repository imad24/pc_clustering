import pandas as pd
import math
import numpy as np

import pandas as pd
import numpy as np

from sklearn import metrics

import matplotlib.pyplot as plt
import copy as cp
import matplotlib.cm as cm
import seaborn as sns


#import statsmodels.api as sm

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

def Clusters_plot(X,labels, inertia = 0,info=["","",""]):
    plt.figure(figsize=(16,7))
    colors = [str(item/255.) for item in labels]
    
    
    
    plt.suptitle("Clustering Method: (%s,%s,%s)"%(info[0],info[1],info[2]),size=14) 
    plt.subplot(1,2,1)
    plt.scatter(X[:,0],X[:,1],cmap ="Paired" ,c=colors)
    plt.xlabel("PCA01")    
    plt.ylabel("PCA02")

#     centroids = np.array(list(set(labels)))/
#     for center in centroids:
#         #get points of each cluster
#         cluster = np.where(labels == center)[0]

#         #determine position and covariance
#         points = X[cluster,0:2]
#         pos = X[center,0:2]
#         cov = np.cov(points, rowvar=False)
#         print(colors[center])
#         rgba = cmap(0.3)
#         #plot Ellipse
#         plot_cov_ellipse(cov, pos, alpha = 0.5,color=rgba,cmap ="Paired")

    plt.subplot(1,2,2)
    plt.scatter(X[:,0],X[:,2],cmap ="Paired" ,c=colors)
    plt.xlabel("PCA01")    
    plt.ylabel("PCA03") 
   
    plt.show(block = True)

def decorate_plot(cols,tick_frequency = 3, rotation = 70,color='lightblue'):
    weeks = np.arange(len(cols))
    for x in weeks: 
        if (x+1)%12 == 0: plt.axvline(x,c=color,linestyle='--')
    plt.xticks(weeks[::tick_frequency], cols[::tick_frequency], rotation = rotation)
    
    
    
def getSSE(samples,centroids):
    return np.sum( (samples-centroids)**2)

def getSilouhaite(samples,labels):
    return metrics.silhouette_score(samples,labels)


from ast import literal_eval as make_tuple

def Cluster_series_plot(data_df,cluster_df,centroid_only= False,headers = None):
    
    list_it = list(range(len(data_df.columns)))
    tick_frequency = 3
    
    clusters = list(set(cluster_df['Cluster']))
    nc = len(clusters)
    
    clusters_array = []
    
    plt.figure(figsize=(15,nc*4))

    if headers:
        nh = len(headers)+2
        #Joinc clustering results with the series
        merged_df = pd.merge(cluster_df,data_df ,  how='left', left_on=headers, right_on = headers)
        for i,c in enumerate(clusters):
            plt.subplot(nc,1,i+1)
            #get only series within the cluster
            df = merged_df[merged_df['Cluster']==c]
            #get the medoid from the first occurence
            medoid  = make_tuple(df["Centroid"].iloc[0])

            plt.title("Cluster %d (%s)): %d product"%(c,medoid,df.shape[0]))
            for index, row in df.iterrows():
                key = (row["Product"],row["Client"])
                if centroid_only and key!= medoid: continue
                plt.plot(list(row)[nh:],label = index)
            plt.xticks(list_it[nh::tick_frequency], list(data_df.columns)[nh::tick_frequency], rotation = 70)
            clusters_array += [[i,key]]
            #plt.legend(loc=0)
    else:
        for i,c in enumerate(clusters):
            plt.subplot(nc,1,i+1)
            product_keys = list(cluster_df[cluster_df['Cluster']==c].iloc[:,0])
            medoid  = cluster_df[cluster_df['Cluster']==c].iloc[0,2]
            plt.title("Cluster %d (%s)): %d product"%(c,medoid,len(product_keys)))
            mask = (data_df['Product'].isin(product_keys))
            df  = data_df[mask]
            for index, row in df.iterrows():
                if centroid_only and row["Product"]!= medoid: continue
                plt.plot(list(row)[1:],label = index)
            plt.xticks(list_it[1::tick_frequency], list(data_df.columns)[1::tick_frequency], rotation = 70)
            clusters_array += [[i,product_keys]]
            plt.legend(loc=0)

    plt.tight_layout()
    #plt.show()
    return clusters_array














    # from matplotlib.patches import Ellipse
# def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
#     """
#     Plots an `nstd` sigma error ellipse based on the specified covariance
#     matrix (`cov`). Additional keyword arguments are passed on to the 
#     ellipse patch artist.

#     Parameters
#     ----------
#         cov : The 2x2 covariance matrix to base the ellipse on
#         pos : The location of the center of the ellipse. Expects a 2-element
#             sequence of [x0, y0].
#         nstd : The radius of the ellipse in numbers of standard deviations.
#             Defaults to 2 standard deviations.
#         ax : The axis that the ellipse will be plotted on. Defaults to the 
#             current axis.
#         Additional keyword arguments are pass on to the ellipse patch.

#     Returns
#     -------
#         A matplotlib ellipse artist
#     """
#     def eigsorted(cov):
#         vals, vecs = np.linalg.eigh(cov)
#         order = vals.argsort()[::-1]
#         return vals[order], vecs[:,order]

#     if ax is None:
#         ax = plt.gca()

#     vals, vecs = eigsorted(cov)
#     theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

#     # Width and height are "full" widths, not radius
#     width, height = 2 * nstd * np.sqrt(vals)
#     ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

#     ax.add_artist(ellip)
#     return ellip