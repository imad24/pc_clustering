import pandas as pd
import math
import numpy as np

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import copy as cp
import matplotlib.cm as cm
import seaborn as sns


import statsmodels.api as sm

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

def GetMostCorrelatedTo(X_embedded,component,n=5,absl = True ):
    lx = X_embedded.shape[0]+1
    ly = X_embedded.shape[1]+1
    df_Xpca = pd.DataFrame(X_embedded,index=np.arange(1,lx),columns = np.arange(1,ly))
    return df_Xpca.abs().nlargest(n,component) if absl else df_Xpca.nlargest(n,component)

def Clusters_plot(X,labels, inertia = 0,info=["","",""]):
    plt.figure(figsize=(16,7))
    cmap = cm.get_cmap('Paired')
    colors = [str(item/255.) for item in labels]
    
    centroids = np.array(list(set(labels)))
    
    
    plt.suptitle("Clustering Method: (%s,%s,%s)"%(info[0],info[1],info[2]),size=14) 
    plt.subplot(1,2,1)
    plt.scatter(X[:,0],X[:,1],cmap ="Paired" ,c=colors)
    plt.xlabel("PCA01")    
    plt.ylabel("PCA02")
    
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


from matplotlib.patches import Ellipse
def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip