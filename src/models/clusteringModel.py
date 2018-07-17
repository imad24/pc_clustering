from sklearn.metrics import silhouette_score
from external import kMedoids
from scipy.spatial.distance import pdist,squareform
from scipy.cluster import hierarchy

import settings
import numpy as np

class ClusteringModel:
    """A class holding a clustering model object, inspired from sklearn structure
    """

    # The models supported by the class
    # TODO: Enhance with other models: HCA, kmeans....
    models = ["kMedoids"]


    @classmethod
    def select_best_k(cls, inertia_grid, silhouette_grid, best = None):
        if (best is None): best = settings.options["best_k"]
        for k in inertia_grid:
            if list(silhouette_grid).index(k) < best:
                return k
    
    @classmethod
    def agg_cut_off(cls,data,method="complete",metric="euclidean"):
        Z = hierarchy.linkage(data, method=method,metric=metric)
        last = Z[-150:, 2]

        acceleration = np.diff(last, 2)  # 2nd derivative of the distances
        acceleration_rev = acceleration[::-1]
        best_ks = np.abs(acceleration_rev).argsort()[::-1]
        k =  best_ks+ 2  # if idx 0 is the max of this we want 2 clusters
        return int(k[0])

    def __init__(self, name, k,init_method):
        if (name not in ClusteringModel.models): 
            raise "Model name not recognized"
        self.name = name
        self.k = k
        self.labels = []  
        self.centroids = []
        self.X = []
        self.distances = []
        self.init_method = init_method

    def fit(self,X,k=None,weights = None, init_method = None):
        if (init_method is None): init_method = self.init_method
        if (k is None): k = self.k
        self.X = X
        # TODO: Add weights to distances
        self.distances = self.get_distances(X)
        if (self.name) == "kMedoids":
            self.labels, self.centroids = kMedoids.cluster(self.distances,k=k,init=init_method,X=self.X)

    def get_SSE(self):
        return np.sum( (self.X-self.X[self.labels])**2)

    def get_inertia(self):
        return np.sqrt(self.get_SSE()/len(self.labels))

    def get_silhouette(self):
        return silhouette_score(self.X,self.labels)

    def get_distances(self,X,distance = "euclidean"):
        return squareform(pdist(X, distance))


    def grid_search(self,X,weights = None,k_values = [],order=0):
        inertia = []
        silhouette = []

        K_values = np.array(k_values)

        for k in K_values:
            self.fit(X=X,k = k, weights = weights)
            silhouette.append(self.get_silhouette())
            sse = self.get_SSE()
            inertia.append(np.sqrt(sse/len(self.labels)))

        acc = np.diff(inertia, 2)
        best_ks = acc.argsort()[::-1]
        i = best_ks+ 1
        inertia_grid = np.array(K_values[i])

        sil = np.array(silhouette)
        best_ks = sil.argsort()[::-1]
        silhouette_grid = np.array(K_values[best_ks])

        return inertia_grid, silhouette_grid