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
    sdge        -- spatial expression matrix over locations (locations x genes)
    locations   -- spatial coordinates (locations x dimensions)
    n_neighbors -- defining the size of neighborhood for checking autocorrelation
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

def compute_random_coupling(p, q, epsilon):
    """
    Computes a random coupling based on:

    KL-Proj_p,q(K) = argmin_T <-\epsilon logK, T> -\epsilon H(T)
    where T is a couping matrix with marginal distributions p, and q, for rows and columns, respectively

    This is solved with a Bregman Sinkhorn computation
    p       -- marginal distribution of rows
    q       -- marginal distribution of columns
    epsilon -- entropy coefficient
    """
    num_cells = len(p)
    num_locations = len(q)
    K = np.random.rand(num_cells, num_locations)
    C = -epsilon * np.log(K)
    return sinkhorn(p, q, C, epsilon)

def get_cell_entropy(T, norm=True):
    """
    Compute entropy of normalized transport probabilities for each cell
    T -- transport matrix (cells x locations)
    """
    if len(T.shape) == 1:
        T = T.reshape((1, -1))
    T = (T.T / T.sum(1)).T if norm else T
    return (-T * np.log2(T)).sum(axis=1)

def correlation_random_markers(tissue, with_atlas=True, with_repeats=False,
                             alpha_linears=[], epsilons=[], num_markerss=[], repeats=10):
    """
    Reports the Pearson correlation with the atlas and with repeats sampling different markers
    with_atlas                            -- compute correlation with atlas
    with_repeats                          -- compute correlation with repeats sampling different markers (consistency)
    alpha_linears, epsilons, num_markerss -- lists of parameters that can vary
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
