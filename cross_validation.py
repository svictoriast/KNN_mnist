import numpy as np
from collections import defaultdict
from nearest_neighbors import KNNClassifier


def kfold(n, n_folds=5):
    ans = []
    folds_sizes = [n // n_folds + 1 for _ in range(n % n_folds)]
    folds_sizes += [n // n_folds for _ in range(n_folds - n % n_folds)]
    ind_range = np.arange(n)
    test_right = 0
    test_left = 0
    train_left = n
    for i in range(n_folds):
        test_left += folds_sizes[i]
        test = ind_range[test_right:test_left]
        train = np.concatenate([ind_range[:test_right], ind_range[test_left:train_left]])
        ans.append((train, test))
        test_right = test_left
        if n_folds - i == 1:
            train_left = test_right
    return ans


def accuracy(y_true, y_pred):
    correct_labels = np.sum(y_true == y_pred)
    return correct_labels / len(y_true)


def cv_predict(model, k_neighbours, distances=None):
    y_pred = np.array(k_neighbours.shape[0])
    classes = model.y[k_neighbours.astype(int)]
    unique_labels = np.unique(model.y)
    if not model.weights:
        all_labels = np.sum((classes[:, :, None] == unique_labels), axis=1)
    else:
        eps = 1e-5
        weights = 1 / (eps + distances)
        all_labels = np.sum(weights[:, :, None] * (classes[:, :, None] == unique_labels), axis=1)
    y_pred = unique_labels[np.argmax(all_labels, axis=1)]
    return y_pred


def knn_cross_val_score(X, y, k_list, score='accuracy', cv=None, **kwargs):
    d = defaultdict(lambda: np.array([]))
    if cv is None:
        cv = kfold(X.shape[0], 5)
    model = KNNClassifier(k=k_list[-1], **kwargs)
    for cv_ind in cv:
        model.fit(X[cv_ind[0]], y[cv_ind[0]])
        distances = []
        if model.weights:
            distances, k_neighbors = model.find_kneighbors(X[cv_ind[1]], model.weights)
        else:
            k_neighbors = model.find_kneighbors(X[cv_ind[1]], model.weights)
        for k in k_list[::-1]:
            if len(distances):
                distances = distances[:, :k]
            k_neighbors = k_neighbors[:, :k]
            cv_pred = cv_predict(model, k_neighbors, distances)
            score_value = accuracy(y[cv_ind[1]], cv_pred)
            if k in d:
                d[k] = np.append(d[k], score_value)
            else:
                d[k] = [score_value]
    return d
