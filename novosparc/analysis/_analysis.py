import numpy as np
import pysal.lib
from pysal.explore.esda.moran import Moran
import time
from multiprocessing import Pool
import os

def pool_wrapper(args):
    mi_ = Moran(*args)
    return mi_.I, mi_.p_norm

def morans(sdge, gene_names, locations, folder, selected_genes=None, num_important_genes=10):
    """Calculates Moran's I metric to select for spatially informative genes"""

    start_time = time.time()

    if selected_genes is not None:
        selected_genes = np.asarray(selected_genes)
        gene_indices = np.nonzero(np.in1d(gene_names, selected_genes))[0]
        sdge = sdge[gene_indices, :]
        gene_names = selected_genes

    num_genes = sdge.shape[0]

    print ('Setting up Morans I analysis for %i genes...' % num_genes, end='', flush=True)
 
    expression = (sdge.T/np.sum(sdge,axis=1))

    #initate moran's I and pval with zeros
    mi = np.zeros(num_genes)
    mi_pval = np.zeros(num_genes)

    wknn3 = pysal.lib.weights.KNN(locations, k=8)

    #Iterate through the genes
    gene_vals = expression.T.tolist()
    knn_list = [wknn3] * len(gene_vals)

    pool = Pool(32)
    mi_,mi_pval_ = zip(*pool.map(pool_wrapper, zip(gene_vals, knn_list)))
    mi = np.asarray(mi_)
    mi_pval = np.asarray(mi_pval_)

    mi[np.isnan(mi)] = -np.inf

    important_gene_ids = np.argsort(mi)[::-1][:num_important_genes]
    important_gene_names = gene_names[important_gene_ids]

    results =  np.column_stack((gene_names, mi, mi_pval))
    print ('done (', round(time.time()-start_time, 2), 'seconds )')

    np.savetxt(os.path.join(folder, str(num_genes) + '_genes_'
        + str(locations.shape[0]) + '_locations_' + 'morans.txt'), results, delimiter="\t", fmt="%s")

    return important_gene_names

