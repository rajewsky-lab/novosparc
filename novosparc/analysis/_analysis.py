import os
import time
import copy
import random
import numpy as np
import pandas as pd
from ot.bregman import sinkhorn
import scipy.stats as stats
from multiprocessing import Pool
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import squareform, pdist
from esda.moran import Moran
from pysal.lib import weights
from shapely.geometry import Point
import geopandas as gpd
from scipy.stats import pearsonr


def get_moran_pvals(sdge, locations, n_neighbors=8):
    """
    Computes Moran's I autocorrelation and its corresponding p-values (considering a normal, one-tailed test)
    """
    cols = ['X', 'Y', 'Z']
    d = locations.shape[1]
    df = pd.DataFrame({cols[i]: locations[:, i] for i in np.arange(d)})
    df['coords'] = df[cols[:d]].apply(Point, axis=1)
    gdf = gpd.GeoDataFrame(df, geometry='coords')
    w = weights.distance.KNN.from_dataframe(gdf, k=n_neighbors)

    I = []
    pvals = []
    for g in np.arange(sdge.shape[1]):
        y = sdge[:,g]
        mi = Moran(y, w, two_tailed=False)
        I.append(mi.I)
        pvals.append(mi.p_norm)

    return np.array(I), np.array(pvals)

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

def get_cell_entropy(A, norm=True):
    if len(A.shape) == 1:
        A = A.reshape((1, -1))
    A = (A.T / A.sum(1)).T if norm else A
    return (-A * np.log2(A)).sum(axis=1)

def correlation_random_markers(tissue, with_atlas=True, with_repeats=False,
                             alpha_linears=[], epsilons=[], num_markerss=[], repeats=10):
    """
    Reports the Pearson correlation with the atlas
    """

    stissue = copy.deepcopy(tissue)
    atlas_matrix = stissue.atlas_matrix
    markers_to_use = stissue.markers_to_use
    if (not np.any(atlas_matrix)) or (not np.any(markers_to_use)):
        print('Tissue object is missing a reference atlas')
    nruns = len(alpha_linears) * len(epsilons) * len(num_markerss) * repeats
    print('Running %d computations. This will take a while...' % nruns)
    num_markers_all = atlas_matrix.shape[1]

    res = []
    params = ['alpha_linear', 'epsilon', 'num_markers']
    for _ in np.arange(repeats):
        for num_markers in num_markerss:
            # random selection of markers
            idx = np.random.choice(num_markers_all, num_markers)
            smarkers_to_use = markers_to_use[idx]
            satlas_matrix = atlas_matrix[:, idx]
            stissue.setup_linear_cost(smarkers_to_use, satlas_matrix)

            # reconstruct with various alphas
            for alpha_linear in alpha_linears:
                for epsilon in epsilons:
                    stissue.reconstruct(alpha_linear=alpha_linear, epsilon=epsilon)

                    # measure correlation with atlas
                    res_s = {'alpha_linear': alpha_linear,
                             'epsilon': epsilon,
                             'num_markers': num_markers}
                    if with_atlas:
                        corr_atlas = pearsonr(stissue.sdge[markers_to_use, :].T.flatten(),
                                              atlas_matrix.flatten())[0]
                        res_s['Pearson correlation'] = corr_atlas

                    if with_repeats:
                        res_s['sdge'] = stissue.sdge.flatten()
                    res.append(res_s)

    df = pd.DataFrame(res)

    # compute self-consistency
    df_corr_repeats = None
    df_corr_atlas = None
    if with_repeats:
        res_corr_repeats = []
        groups = df.groupby(params).groups
        for param, idx in groups.items():
            for ii, i in enumerate(idx):
                for ij, j in enumerate(idx):
                    if ij <= ii:
                        continue
                    corr_repeat = pearsonr(df.loc[i]['sdge'], df.loc[j]['sdge'])[0]
                    res_corr_repeats.append({'alpha_linear': param[0],
                                             'epsilon': param[1],
                                             'num_markers': param[2],
                                             'Pearson correlation': corr_repeat})
        df_corr_repeats = pd.DataFrame(res_corr_repeats)

    if with_atlas:
        df_corr_atlas = df[params + ['Pearson correlation']]  # with atlas

    return df_corr_atlas, df_corr_repeats
