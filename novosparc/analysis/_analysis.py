import os
import time
import random
import numpy as np
import pandas as pd
from ot.bregman import sinkhorn
import scipy.stats as stats
from multiprocessing import Pool
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import squareform, pdist


def get_thr_neighbor(geometry, n_neighbors=5):
    """
    Returns the approximate diameter around a single cell with n neighbors
    Ac = dx * dy /ncells - avg area per cell
    sqrt(Ac * n) - diameter with n pts
    """
    xy = geometry[['x', 'y']].values if isinstance(geometry, pd.DataFrame) else geometry
    d = xy.shape[1]
    vol = 1
    for i in np.arange(d):
        dx = xy[:, i].max() - xy[:, i].min()
        if dx == 0:
            d -= 1
        else:
            vol = vol * dx
    # vol / locations - avg vol per location
    # then the vol of k neighbors is * neighbors
    # to get the diameter, take to the power 1/d
    # for radius, divide by 2
    nlocations = len(np.unique(xy, 1)[0])
    min_dist = np.min(pdist(xy))
    thr = max(min_dist, np.power(vol / nlocations * n_neighbors, 1 / d) / 2)

    return thr


def get_morans(xy, X, radius=None, n_neighbors=5, exclude_self=False, W=None):
    if W is None:
        radius = get_thr_neighbor(xy, n_neighbors=n_neighbors) if radius is None else radius
        cell_dists = squareform(pdist(xy))
        if exclude_self:
            np.fill_diagonal(cell_dists, np.Inf)
        W = (cell_dists <= radius).astype(int)

    n, g = X.shape
    X = np.array(X)
    S0 = np.sum(W)

    Z = (X - np.mean(X, 0))
    sZs = (Z ** 2).sum(0)
    I = n / S0 * (Z.T @ (W @ Z)) / sZs
    return np.diagonal(I)

def get_moran_pvals(sdge, locations, radius=None, verbose=False, npermut=100, n_neighbors=8):
    """
    Calculates the spatial correlation given a threshold radius of neighbors
    """
    nns = kneighbors_graph(locations, 8, mode='connectivity', include_self=False)
    W = nns.toarray()
    X = sdge
    xy = locations
    n, g = X.shape
    X = np.array(X)
    pI = np.zeros((npermut, g))
    idx = np.arange(n)

    for i in np.arange(npermut):
        pidx = np.random.permutation(idx)
        pI[i, :] = get_morans(xy, X[pidx, :], W=W).reshape((1, -1))

    I = get_morans(xy, X, W=W)
    pvals = np.mean(I <= pI, 0)

    return I, pvals

def change_matrix_same_sum(T, times=10):
    """Makes a change to a matrix while:
        1. Leaving the sum of rows and columns constant
        2. Maintaining non-negative values. """
    p = np.sum(T, axis=1)
    q = np.sum(T, axis=0)

    n, m = T.shape
    size = int(np.floor(min(n, m)/2.0) * 2)
    min_change = 0 # 1 / (n * m) * 10e-3

    for _ in np.arange(times):
        # sample rows and cols of influence
        samp_rows = random.sample(list(np.arange(n)), size) # samp size?
        samp_cols = random.sample(list(np.arange(m)), size)
        sel_rows = np.repeat(samp_rows, size)
        sel_cols = np.tile(samp_cols, size)

        # sample change
        subT = T[sel_rows, sel_cols].reshape((size, size))
        max_change = np.min(subT)
        # so numbers don't get too small
        # if max_change < min_change:
        #     continue
        change = np.random.uniform(low=min_change, high=max_change)

        # update interlaced grid
        subT[0::2, 0::2] += change
        subT[1::2, 1::2] += change
        subT[1::2, 0::2] -= change
        subT[0::2, 1::2] -= change
        T[sel_rows, sel_cols] = subT.flatten()

    # print("Row sums squared error: %f " % np.sum(np.square(np.sum(T, axis=1) - p)))
    # print("Column sums squared error: %f " % np.sum(np.square(np.sum(T, axis=0) - q)))

    return T


def compute_random_coupling(p, q, epsilon):
    """
    Computes a random coupling based on:

    KL-Proj_p,q(K) = argmin_T <-\epsilon logK, T> -\epsilon H(T)
    where T is a couping matrix with marginal distributions p, and q, for rows and columns, respectively

    This is solved with a Bregman Sinkhorn computation
    """
    num_cells = len(p)
    num_locations = len(q)
    K = np.random.rand(num_cells, num_locations)
    C = -epsilon * np.log(K)
    return sinkhorn(p, q, C, epsilon)
