import numpy as np


def euclidean_distance(X, Y):
    sqrd_X = (X ** 2).sum(axis=1)
    sqrd_Y = (Y ** 2).sum(axis=1)
    euclidean = np.sqrt(np.add.outer(sqrd_X, sqrd_Y) - 2 * X @ Y.T)
    return euclidean


def cosine_distance(X, Y):
    norm_x = np.sqrt((X ** 2).sum(axis=1))
    norm_y = np.sqrt((Y ** 2).sum(axis=1))
    cosine = (X @ Y.T) / (np.outer(norm_x, norm_y) + 1e-9)
    return 1 - cosine
