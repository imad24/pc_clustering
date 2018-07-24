from sklearn.metrics import silhouette_score
from external import kMedoids
from scipy.spatial.distance import pdist,squareform
from scipy.cluster import hierarchy

import settings
import numpy as np


class classifierModel:

    le_encoder = {}
    ohe_encoder = {}
    scaler = None
    model = None
    non_categorical = []
    categorical = []

    def __init__(self):

        pass

    def fit(self,X):
        self.model.fit(X)

    def predict(self,X):
        self.model.predict(X)
