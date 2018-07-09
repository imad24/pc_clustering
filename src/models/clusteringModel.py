from sklearn.metrics import silhouette_score
from external import kMedoids
from scipy.spatial.distance import pdist,squareform

import numpy as np

class ClusteringModel:

    models = ["kMedoids"]


    @staticmethod
    def select_best_k(inertia_grid,silhouette_grid):
        for k in inertia_grid:
            if list(silhouette_grid).index(k)<5:
                return k

    def __init__(self, name, k):
        if (name not in ClusteringModel.models): 
            raise "Model name not recognized"
        self.name = name
        self.k = k
        self.labels = []  
        self.centroids = []
        self.X = []
        self.distances = []

    def fit(self,X,k=None,weights = None):
        if (k is None): k = self.k
        self.X = X
        # TODO: Add weights to distances
        self.distances = self.get_distances(X)
        if (self.name) == "kMedoids":
            self.labels, self.centroids = kMedoids.cluster(self.distances,k=k)

    def getSSE(self):
        return np.sum( (self.X-self.X[self.labels])**2)

    def getInertia(self):
        return np.sqrt(self.getSSE()/len(self.labels))

    def getSilhouette(self):
        return silhouette_score(self.X,self.labels)

    def get_distances(self,X,distance = "euclidean"):
        return squareform(pdist(X, distance))


    def grid_search(self,X,weights = None,k_values = [],order=0):
        inertia = []
        silhouette = []

        K_values = np.array(k_values)

        for k in K_values:
            self.fit(X=X,k = k,weights = weights)
            silhouette.append(self.getSilhouette())
            sse = self.getSSE()
            inertia.append(np.sqrt(sse/len(self.labels)))

        acc = np.diff(inertia, 2)
        best_ks = acc.argsort()[::-1]
        i =  best_ks+ 2
        inertia_grid = np.array(K_values[i])

        sil = np.array(silhouette)
        best_ks = sil.argsort()[::-1]
        silhouette_grid = np.array(K_values[best_ks])

        return inertia_grid, np.array(silhouette_grid)