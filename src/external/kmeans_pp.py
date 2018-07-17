import numpy as np
import random

class KPlusPlus:
    def __init__(self,X,k):
        self.X = X
        self.k = k
        self.centers = []
        self.mu = []

    def _dist_from_centers(self):
        cent = self.mu
        X = self.X
        D2 = np.array([min([np.linalg.norm(x-c)**2 for c in cent]) for x in X])
        self.D2 = D2
 
    def _choose_next_center(self):
        self.probs = self.D2/self.D2.sum()
        self.cumprobs = self.probs.cumsum()
        r = random.random()
        ind = np.where(self.cumprobs >= r)[0][0]
        return(self.X[ind],ind)
 
    def init_centers(self):
        #choose first center at random
        first = random.randint(0,self.X.shape[0])

        #store the center and its index
        self.mu.append(self.X[first,:])
        self.centers.append(first)

        #choose next centers according to probability
        while len(self.mu) < self.k:
            self._dist_from_centers()
            center,ind = self._choose_next_center()
            self.mu.append(center)
            self.centers.append(ind)
        return np.array(self.centers)