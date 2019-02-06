from __future__ import print_function

###########
# imports #
###########

import numpy as np
from sklearn import manifold, datasets
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import cdist
from scipy.cluster import hierarchy
from scipy.stats import pearsonr
import ot
import networkx as nx
from textwrap import wrap
import time
import sys
import zipfile

#############
# functions #
#############

def log_normalize_dge(dge):
    """Log-normalize raw counts if needed."""
    return np.round(np.log2(150000 * np.divide(dge, np.sum(dge, axis=0)) + 1), 2)


def introduce_noise(expression, dropouts=0.1, gaussian=False, mu=0,
                    sigma=0.05, false_positives=False):
    """Introduces noise in the data in a variety of ways.
    dropouts        -- % of zeros to be introduced in every gene (takes values in [0, 1])
    gaussian        -- boolean, whether to insert Gaussian noise or not
    mu, sigma       -- the mean and standard deviation for the Gaussian noise
    false_positives -- whether to introduce false positives in the data
                       (modelling mixing from ambient RNA)"""
    num_cells, num_genes = expression.shape
    noisy_expression = np.copy(expression)
    if dropouts > 0:
        for gene in range(num_genes):
            cells_on = np.where(expression[:, gene] > 0)[0]
            noisy_expression[np.random.choice(cells_on, int(dropouts*len(cells_on)),
                                              replace=False), gene] = 0

    if gaussian:
        for gene in range(num_genes):
            noisy_expression[:, gene] += np.random.normal(mu, sigma, num_cells)
    return noisy_expression
                                                                                
    
def pca(expression, n_components=2):
    """PCA from sklearn.
    expression -- the gene expression data, genes as columns and cells as rows."""
    pca = PCA(n_components=n_components)
    pca.fit(expression.T)
    return pca


def identify_highly_variable_genes(expression, low_x=1, high_x=8, low_y=1, do_plot=True):
    """Identify the highly variable genes (follows the Seurat function).
    expression -- the DGE with cells as columns
    low_x      -- threshold for low cutoff of mean expression
    high_x     -- threshold for high cutoff of mean expression
    low_y      -- threshold for low cutoff of scaled dispersion."""
 
    mean_val = np.log(np.mean(np.exp(expression)-1, axis=1) + 1)
    gene_dispersion = np.log(np.var(np.exp(expression) - 1, axis=1) / np.mean(np.exp(expression) - 1))
    bins = np.arange(1, np.ceil(max(mean_val)), step=0.5)
    binned_data = np.digitize(mean_val, bins)
    # This should be written more efficiently
    gd_mean = np.array([])
    gd_std = np.array([])
    for bin in np.unique(binned_data):
        gd_mean = np.append(gd_mean, np.mean(gene_dispersion[binned_data == bin]))
        gd_std = np.append(gd_std, np.std(gene_dispersion[binned_data == bin]))
    gene_dispersion_scaled = (gene_dispersion - gd_mean[binned_data]) / (gd_std[binned_data] + 10e-8)
    genes = np.intersect1d(np.where(mean_val >= low_x), np.where(mean_val <= high_x))
    genes = np.intersect1d(genes, np.where(gene_dispersion_scaled >= low_y))

    if do_plot:
        col_genes = np.zeros(len(expression))
        col_genes[genes] = 1
        plt.figure()
        plt.scatter(mean_val, gene_dispersion_scaled, s=2, c=col_genes)
        plt.savefig('output/high_variable_genes.png')

    return genes
