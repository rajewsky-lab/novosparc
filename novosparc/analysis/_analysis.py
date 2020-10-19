import os
import time
import numpy as np
import pandas as pd
import scipy.stats as stats
from multiprocessing import Pool
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import squareform, pdist

def calc_pval(expression, w, I):
    N = len(expression)
    W = np.sum(w)
    EI = -1. / (N - 1)
    z = expression - np.mean(expression)

    S1 = (w + w.T) * (w + w.T)
    S1 = np.sum(S1) / 2.

    S2 = np.sum(np.array(w.sum(1) + w.sum(0).transpose()) ** 2)

    S3_num = 1. / N * np.sum(z ** 4)
    S3_denom = ((1. / N) * np.sum(z ** 2)) ** 2
    S3 = S3_num / S3_denom

    S4 = (N ** 2 - 3 * N + 3) * S1 - N * S2 + 3 * (W ** 2)

    S5 = (N ** 2 - N) * S1 - 2 * N * S2 + 6 * (W ** 2)

    var = (N * S4 - S3 * S5) / ((N - 1) * (N - 2) * (N - 3) * (W ** 2)) - EI ** 2

    z_norm = (I - EI) / (var ** (1 / 2))

    if z_norm > 0:
        p_norm = 1 - stats.norm.cdf(z_norm)
    else:
        p_norm = stats.norm.cdf(z_norm)

    return p_norm

def Moran(expression, weights):
    N = len(expression)
    W = np.sum(weights)
    z = expression - np.mean(expression)

    numerator = np.sum(weights * np.outer(z, z))
    denominator = np.sum(z * z)

    I = (N / W) * (numerator / denominator)

    p_norm = calc_pval(expression, weights, I)


    return I, p_norm

def pool_wrapper(args):
    I, p_norm = Moran(*args)
    return I, p_norm

def morans(sdge, gene_names, locations, folder=None, selected_genes=None, num_important_genes=10):
    """Calculates Moran's I metric to select for spatially informative genes"""

    start_time = time.time()

    if selected_genes is not None:
        selected_genes = np.asarray(selected_genes)
        gene_indices = np.nonzero(np.in1d(gene_names, selected_genes))[0]
        sdge = sdge[gene_indices, :]
        gene_names = selected_genes

    num_genes = sdge.shape[0]

    print('Setting up Morans I analysis for %i genes...' % num_genes, end='', flush=True)

    expression = (sdge.T / np.sum(sdge, axis=1))

    # initate moran's I and pval with zeros
    mi = np.zeros(num_genes)
    mi_pval = np.zeros(num_genes)

    nns = kneighbors_graph(locations, 8, mode='connectivity', include_self=False)
    wknn3 = nns.toarray()

    # Iterate through the genes
    gene_vals = expression.T.tolist()
    knn_list = [wknn3] * len(gene_vals)

    pool = Pool(32)
    mi_, mi_pval_ = zip(*pool.map(pool_wrapper, zip(gene_vals, knn_list)))
    mi = np.asarray(mi_)
    mi_pval = np.asarray(mi_pval_)

    mi[np.isnan(mi)] = -np.inf

    important_gene_ids = np.argsort(mi)[::-1][:num_important_genes]
    important_gene_names = gene_names[important_gene_ids]

    results = np.column_stack((gene_names, mi, mi_pval))
    print('done (', round(time.time() - start_time, 2), 'seconds )')

    if folder is not None:
        np.savetxt(os.path.join(folder, str(num_genes) + '_genes_'
                                + str(locations.shape[0]) + '_locations_' + 'morans.txt'), results, delimiter="\t",
                   fmt="%s")

    return important_gene_names





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

def get_moran_pvals(xy, X, radius=None, verbose=False, npermut=100, n_neighbors=5, exclude_self=False):
    """
    Calculates the spatial correlation given a threshold radius of neighbors
    """
    radius = get_thr_neighbor(xy, n_neighbors=n_neighbors) if radius is None else radius

    cell_dists = squareform(pdist(xy))
    if exclude_self:
        np.fill_diagonal(cell_dists, np.Inf)
    W = (cell_dists <= radius).astype(int)

    if verbose:
        print("Mean number of neighbors : ", np.mean(np.sum(W, 0)))

    n, g = X.shape
    X = np.array(X)
    pI = np.zeros((npermut, g))
    idx = np.arange(n)

    for i in np.arange(npermut):
        pidx = np.random.permutation(idx)
        pI[i, :] = morans(xy, X[pidx, :], W=W).reshape((1, -1))

    I = morans(xy, X, W=W)
    pvals = np.mean(I <= pI, 0)

    return I, pvals
