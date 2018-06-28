from sklearn.metrics import silhouette_score
from external import kMedoids
from scipy.spatial.distance import pdist,squareform

import numpy as np

class ClusteringModel:

    models = ["kMedoids"]

    def __init__(self, name, k):
        if (name not in ClusteringModel.models): 
            raise "Model name not recognized"
        self.name = name
        self.k = k
        self.labels = []  
        self.centroids = []
        self.X = []
        self.distances = []

    def fit(self,X,k,weights = None):
        if (k is None): k = self.k
        self.X = X
        # TODO: Add weights to distances
        self.distances = self.get_distances(X)
        if (self.name) == "kMedoids":
            self.labels, self.centroids = kMedoids.cluster(self.distances,k=k)

    def getSSE(self):
        return np.sum( (self.X[self.labels]-self.X[self.centroids])**2)

    def getInertia(self):
        return np.sqrt(self.getSSE()/len(self.labels))

    def getSilhouette(self):
        return silhouette_score(self.X,self.labels)

    def get_distances(self,X,distance = "euclidean"):
        return squareform(pdist(X, distance))


    def fit_select(self,X,weights = None,k_values = []):
        for k in k_values:
            self.fit(X=X,k = k,weights = weights)