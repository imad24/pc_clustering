import numpy as np
import random

from sklearn.decomposition import PCA
from external import skmeans_pp

def cluster(distances, k=3,init="random",X=None):
    """Performs a Kmedoids clustering using distances matrix around k cluster

    
    Arguments:
        distances {2d numpy array} -- array of distances between each point (square form)
    
    Keyword Arguments:
        k {int} -- number of clusters (default: {3})
        init {str} -- Method for initialization: kmeans++, PCA or random (default: {"random"})
        X {2d numpy array} -- the data matrix required when init method is not None (default: {None})
    
    Raises:
        ValueError -- [description]
        ValueError -- [description]
        ValueError -- [description]
    
    Returns:
        [type] -- [description]
    """

    m = distances.shape[0] # number of points

    #medoids
    curr_medoids = np.array([-1]*k)

    if X is None and init != "random": 
        raise ValueError("X data must be provided when PCA init is used.")
    if (X.shape[0] != distances.shape[0]) and (init != "random"):
        raise ValueError("Distance matrix and X matrix have different number of samples %d and %d "%(distances.shape[0],X.shape[0]))
    if init == "PCA":
        #Calculate the PCA transformation
        X_pca = PCA(n_components=k).fit_transform(X)
        #keep as initial medoids the K samples that are the most correlated to the k components
        curr_medoids = np.argmax(X_pca,axis=0)
    elif init=="kmeans++":
        skpp = skmeans_pp.KPlusPlus(X,k)
        curr_medoids = skpp.sk_init_centers()
    elif init=="ckmeans++":
        skpp = skmeans_pp.KPlusPlus(X,k)
        curr_medoids = skpp.init_centers()
    elif init == "random":
    # Pick k random medoids.
        while not len(np.unique(curr_medoids)) == k:
            curr_medoids = np.array([random.randint(0, m - 1) for _ in range(k)])
    else:
        raise ValueError("Please set a valid init method name")

    old_medoids = np.array([-1]*k) # Doesn't matter what we initialize these to.
    new_medoids = np.array([-1]*k)

    # Until the medoids stop updating, do the following:
    while not ((old_medoids == curr_medoids).all()):
        # Assign each point to cluster with closest medoid.
        clusters = assign_points_to_clusters(curr_medoids, distances)

        # Update cluster medoids to be lowest cost point. 
        for curr_medoid in curr_medoids:
            cluster = np.where(clusters == curr_medoid)[0]
            new_medoids[curr_medoids == curr_medoid] = compute_new_medoid(cluster, distances)

        old_medoids[:] = curr_medoids[:]
        curr_medoids[:] = new_medoids[:]

    return clusters, curr_medoids

def assign_points_to_clusters(medoids, distances):
    distances_to_medoids = distances[:,medoids]
    clusters = medoids[np.argmin(distances_to_medoids, axis=1)]
    clusters[medoids] = medoids
    return clusters

def compute_new_medoid(cluster, distances):
    mask = np.ones(distances.shape)
    mask[np.ix_(cluster,cluster)] = 0.
    cluster_distances = np.ma.masked_array(data=distances, mask=mask, fill_value=10e9)
    costs = cluster_distances.sum(axis=1)
    return costs.argmin(axis=0, fill_value=10e9)
