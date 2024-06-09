import numpy as np
import sklearn.neighbors
from math import ceil
from distances import cosine_distance, euclidean_distance


class KNNClassifier:
    def __init__(self, k=5, strategy='brute', metric='euclidean', weights=False, test_block_size=1200):
        self.k = k
        self.strategy = strategy
        self.metric = cosine_distance if metric == 'cosine' else euclidean_distance
        self.weights = weights
        self.test_block_size = test_block_size
        self.X_train = None
        self.y = None
        self.NN = None
        if strategy != 'my_own':
            if strategy == 'kd_tree' or strategy == 'ball_tree':
                self.NN = sklearn.neighbors.NearestNeighbors(n_neighbors=k, algorithm=strategy, metric='euclidean')
            else:
                self.NN = sklearn.neighbors.NearestNeighbors(n_neighbors=k, algorithm=strategy, metric=metric)

    def fit(self, X, y):
        self.X_train = X
        self.y = y
        if self.strategy != 'my_own':
            self.NN.fit(X)

    def get_blocks(self, X, return_distance):
        ans = [[], []]
        dist_mtrx = self.metric(X, self.X_train)
        ans[1] = np.argpartition(dist_mtrx, kth=np.arange(self.k), axis=1)[:, :self.k].astype(int)
        if return_distance:
            ans[0] = dist_mtrx[np.arange(dist_mtrx.shape[0])[:, None], ans[1]]
        return ans

    def find_kneighbors(self, X, return_distance):
        ind_ans = np.zeros((X.shape[0], self.k))
        dist_ans = []
        if return_distance:
            dist_ans = np.zeros((X.shape[0], self.k))
        num_of_blocks = ceil(X.shape[0] / self.test_block_size)
        extra_blocks = X.shape[0] % self.test_block_size
        r_bndry = extra_blocks if(num_of_blocks == 1 and extra_blocks) \
            else self.test_block_size
        l_bndry = 0
        for i in range(num_of_blocks):
            if self.strategy == 'my_own':
                dist_ans[l_bndry:r_bndry], ind_ans[l_bndry:r_bndry] = \
                    self.get_blocks(X[l_bndry:r_bndry], return_distance)
            else:
                if return_distance:
                    dist_ans[l_bndry:r_bndry], ind_ans[l_bndry:r_bndry] = \
                        self.NN.kneighbors(X[l_bndry:r_bndry], self.k, return_distance)
                else:
                    ind_ans[l_bndry:r_bndry] = self.NN.kneighbors(X[l_bndry:r_bndry], self.k, return_distance)
            l_bndry += self.test_block_size
            if i == num_of_blocks - 2 and extra_blocks:
                r_bndry += extra_blocks
            else:
                r_bndry += self.test_block_size
        if return_distance:
            return dist_ans, ind_ans
        return ind_ans

    def predict(self, X):
        y_pred = np.array(X.shape[0])
        if not self.weights:
            k_neighbours = self.find_kneighbors(X, self.weights)
            classes = self.y[k_neighbours.astype(int)]
            y_pred = np.array([np.argmax(np.bincount(row)) for row in classes])
        else:
            distances, k_neighbours = self.find_kneighbors(X, self.weights)
            eps = 1e-5
            weights = 1 / (eps + distances)
            unique_labels = np.unique(self.y)
            classes = self.y[k_neighbours.astype(int)]
            weighted_classes = np.sum(weights[:, :, None] * (classes[:, :, None] == unique_labels), axis=1)
            y_pred = unique_labels[np.argmax(weighted_classes, axis=1)]
        return y_pred
