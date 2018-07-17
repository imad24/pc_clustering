import numpy as np
import random

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.extmath import row_norms, stable_cumsum


class KPlusPlus:
    def __init__(self, data, k, d=2):
        self.data = data
        self.n = data.shape[0]  # number of data points
        self.d = data.shape[1]  # dimensionality of data
        self.d2 = 2  # returned dimensionality of data (always 2)
        self.k = k
        self.centers = np.zeros((self.k))
        self.centers_id = np.zeros((self.k))



    def sk_init_centers(self):
            """Init n_clusters seeds according to k-means++
            Parameters
            -----------
            X : array or sparse matrix, shape (n_samples, n_features)
                The data to pick seeds for. To avoid memory copy, the input data
                should be double precision (dtype=np.float64).
            n_clusters : integer
                The number of seeds to choose
            Notes
            -----
            Selects initial cluster centers for k-mean clustering in a smart way
            to speed up convergence. see: Arthur, D. and Vassilvitskii, S.
            "k-means++: the advantages of careful seeding". ACM-SIAM symposium
            on Discrete algorithms. 2007
            Version ported from http://www.stanford.edu/~darthur/kMeansppTest.zip,
            which is the implementation used in the aforementioned paper.
            """
            X = self.data
            n_clusters = self.k

            n_samples, n_features = X.shape

            # This is what Arthur/Vassilvitskii tried, but did not report
            # specific results for other than mentioning in the conclusion
            # that it helped.
            n_local_trials = 2 + int(np.log(n_clusters))

            centers = np.empty((n_clusters, n_features), dtype=X.dtype)
            centers_id = np.empty((n_clusters), dtype=int)

            # Pick first center randomly
            center_id = np.random.randint(n_samples)
            centers[0] = X[center_id]
            centers_id[0] = center_id

            x_squared_norms = row_norms(X, squared=True)

            # Initialize list of closest distances and calculate current potential
            closest_dist_sq = euclidean_distances(
                centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms,
                squared=True)
            current_pot = closest_dist_sq.sum()

            # Pick the remaining n_clusters-1 points
            for c in range(1, n_clusters):
                # Choose center candidates by sampling with probability proportional
                # to the squared distance to the closest existing center
                rand_vals = np.random.random_sample(n_local_trials) * current_pot
                candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq),
                                                rand_vals)

                # Compute distances to center candidates
                distance_to_candidates = euclidean_distances(
                    X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True)

                # Decide which candidate is the best
                best_candidate = None
                best_pot = None
                best_dist_sq = None
                for trial in range(n_local_trials):
                    # Compute potential when including center candidate
                    new_dist_sq = np.minimum(closest_dist_sq, distance_to_candidates[trial])
                    new_pot = new_dist_sq.sum()

                    # Store result if it is the best local trial so far
                    if (best_candidate is None) or (new_pot < best_pot):
                        best_candidate = candidate_ids[trial]
                        best_pot = new_pot
                        best_dist_sq = new_dist_sq

                # Permanently add best center candidate found in local tries
                centers[c] = X[best_candidate]
                centers_id[c] = best_candidate
                current_pot = best_pot
                closest_dist_sq = best_dist_sq

            return centers_id


    def init_centers(self):  # adapted from scipy implementation
            k=self.k
            if self.data.any():
                if k is not None:
                    if k > self.data.shape[0]:
                        raise ValueError("k too large:" + str(k) + " for datasize:" + str(self.data.shape[0]))
                    else:
                        self.k = k
                elif self.k is not None:
                    k = self.k
                else:
                    raise ValueError("init_centers no k whatsoever defined")
                n_local_trials = 2 + int(np.log(self.k))  # taken from scikit

                self.centers = np.zeros((self.k, self.d))
                center_id = random.randint(0, self.n - 1)  # choose initial center


                self.centers[0] = self.data[center_id]
                self.centers_id[0] = center_id


                # substract current center from all data points
                delta = self.data - self.centers[0]
                # compute L2 norm
                best_dis = np.linalg.norm(delta, axis=1)  # closest so far for all data points
                best_dis = best_dis**2  # square distance
                best_pot = best_dis.sum()  # best SSE
                for c in range(1, self.k):
                    rand_vals = np.random.random(n_local_trials) * best_pot
                    candidate_ids = np.searchsorted(best_dis.cumsum(), rand_vals)
                    # Decide which candidate is the best
                    best_candidate = None
                    best_pot = None
                    for trial in range(n_local_trials):
                        # substract current candidate from all data points
                        delta_curcand = self.data - self.data[candidate_ids[trial]]
                        # compute L2 norm
                        dis_curcand = np.linalg.norm(delta_curcand, axis=1)
                        dis_curcand = dis_curcand**2
                        # take minimum (must be smaller or equal than previous)
                        dis_cur = np.minimum(best_dis, dis_curcand)
                        new_pot = dis_cur.sum()  # resulting potential
                        # Store result if it is the best local trial so far
                        if (best_candidate is None) or (new_pot < best_pot):
                            best_candidate = candidate_ids[trial]
                            best_pot = new_pot
                            cand_dis = dis_cur
                    best_dis=cand_dis
                    self.centers[c]=self.data[best_candidate] # ibook contains vectors in order of placement!
                    self.centers_id[c] = int(best_candidate)
                # self.ierror=errorOf(self.ibook,self.data)
                # return self.ibook
                return self.centers_id.astype(int)
            else:
                raise "no data present"

    


