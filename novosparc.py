from __future__ import print_function

#########
# about #
#########

__version__ = "0.2.2"
__author__ = ["Nikos Karaiskos", "Mor Nitzan"]
__status__ = "beta"
__licence__ = "GPL"
__email__ = ["nikolaos.karaiskos@mdc-berlin.de", "mornitzan@fas.harvard.edu"]

###########
# imports #
###########

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn import manifold, datasets
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import cdist
from scipy.cluster import hierarchy
from scipy.stats import pearsonr
import ot
import networkx as nx
import GW_adjusted as gwa
from textwrap import wrap
import time
import sys


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

    
def construct_target_grid(num_cells):
    """Constructs a rectangular grid. First a grid resolution is randomly
    chosen. grid_resolution equal to 1 implies equal number of cells and
    locations on the grid. The random parameter beta controls how rectangular
    the grid will be -- beta=1 constructs a square rectangle.
    num_cells -- the number of cells in the single-cell data."""

    grid_resolution = int(np.random.randint(1, 2+(num_cells/1000), 1))
    grid_resolution = 2
    num_locations = len(range(0, num_cells, grid_resolution))
    grid_dim = int(np.ceil(np.sqrt(num_locations)))

    beta = round(np.random.uniform(1, 1.5), 1) # controls how rectangular the grid is
    # beta = 1 # set this for a square grid
    x = np.arange(grid_dim * beta)
    y = np.arange(grid_dim / beta)
    locations = np.array([(i, j) for i in x for j in y])

    return locations


def setup_for_OT_reconstruction(dge, locations, num_neighbors=5):
    start_time = time.time()
    print ('Setting up for reconstruction ... ', end='', flush=True)

    # Shortest paths matrices at target and source spaces
    num_neighbors = num_neighbors # number of neighbors for nearest neighbors graph
    A_locations = kneighbors_graph(locations, num_neighbors, mode='connectivity', include_self=True)
    G_locations = nx.from_scipy_sparse_matrix(A_locations)
    sp_locations = nx.floyd_warshall_numpy(G_locations)
    sp_locations[sp_locations > 5] = 5 #set threshold for shortest paths
    A_expression = kneighbors_graph(dge, num_neighbors, mode='connectivity', include_self=True)
    G_expression = nx.from_scipy_sparse_matrix(A_expression)
    sp_expression = nx.floyd_warshall_numpy(G_expression)

    # Set normalized cost matrices based on shortest paths matrices at target and source spaces
    cost_locations = sp_locations / sp_locations.max()
    cost_locations -= np.mean(cost_locations)
    cost_expression = sp_expression / sp_expression.max()
    cost_expression -= - np.mean(cost_expression)

    print ('done (', round(time.time()-start_time, 2), 'seconds )')
    return cost_expression, cost_locations

def setup_for_OT_reconstruction_1d(dge, locations, num_neighbors_source = 5, num_neighbors_target = 2):
    start_time = time.time()
    print ('Setting up for reconstruction ... ', end='', flush=True)

    # Shortest paths matrices at target and source spaces
    num_neighbors_target = num_neighbors_target # number of neighbors for nearest neighbors graph at target
    A_locations = kneighbors_graph(locations, num_neighbors_target, mode='connectivity', include_self=True)
    G_locations = nx.from_scipy_sparse_matrix(A_locations)
    sp_locations = nx.floyd_warshall_numpy(G_locations)
    num_neighbors_source = num_neighbors_source # number of neighbors for nearest neighbors graph at source
    A_expression = kneighbors_graph(dge, num_neighbors_source, mode='connectivity', include_self=True)
    G_expression = nx.from_scipy_sparse_matrix(A_expression)
    sp_expression = nx.floyd_warshall_numpy(G_expression)

    # Set normalized cost matrices based on shortest paths matrices at target and source spaces
    cost_locations = sp_locations / sp_locations.max()
    cost_locations -= np.mean(cost_locations)
    cost_expression = sp_expression / sp_expression.max()
    cost_expression -= - np.mean(cost_expression)

    print ('done (', round(time.time()-start_time, 2), 'seconds )')
    return cost_expression, cost_locations

def find_spatial_archetypes(num_clusters, sdge):
    """Clusters the expression data and finds gene archetypes. Current
    implementation is based on hierarchical clustering with the Ward method.
    Returns the archetypes, the gene sets (clusters) and the Pearson 
    correlations of every gene with respect to each archetype."""
    print ('Finding gene archetypes ... ', flush=True, end='')
    clusters = hierarchy.fcluster(hierarchy.ward(sdge),
                                  num_clusters,
                                  criterion='maxclust')
    arch_comp = lambda x : np.mean(sdge[np.where(clusters == x)[0], :], axis=0)
    archetypes = np.array([arch_comp(xi) for xi in range(1, num_clusters+1)])
    gene_corrs = np.array([])
    for gene in range(len(sdge)):
        gene_corrs = np.append(gene_corrs, pearsonr(sdge[gene, :],
                                                    archetypes[clusters[gene]-1, :])[0])
    print ('done')
    
    return archetypes, clusters, gene_corrs


def get_genes_from_spatial_archetype(sdge, gene_names, archetypes, archetype, pval_threshold=0):
    """Returns a list of genes which are the best representatives of the archetype
    archetypes       -- the archetypes output of find_spatial_archetypes
    archetype        -- a number denoting the archetype
    pvalue_threshold -- the pvalue returned from the pearsonr function"""
    # Classify all genes and return the most significant ones
    all_corrs = np.array([])
    all_corrs_p = np.array([])
    
    for g in range(len(sdge)):
        all_corrs = np.append(all_corrs, pearsonr(sdge[g, :], archetypes[archetype, :])[0])
        all_corrs_p = np.append(all_corrs_p, pearsonr(sdge[g, :], archetypes[archetype, :])[1])
    indices = np.where(all_corrs_p[all_corrs > 0] <= pval_threshold)[0]
    if len(indices) == 0:
        print ('No genes with significant correlation were found at the current p-value threshold.')
        return None
    genes = gene_names[all_corrs > 0][indices]
    
    return genes


def find_spatially_related_genes(sdge, gene_names, archetypes, gene, pval_threshold=0):
    """Given a gene, find other genes which correlate well spatially.
    gene           -- the index of the gene to be queried
    pval_threshold -- the pvalue returned from the pearsonr function"""
    # First find the archetype of the gene
    arch_corrs = np.array([])
    for archetype in range(len(archetypes)):
        arch_corrs = np.append(arch_corrs, pearsonr(sdge[gene, :], archetypes[archetype, :])[0])
    if np.max(arch_corrs) < 0.7:
        print ('No significant correlation between the gene and the spatial archetypes was found.')
        return None
    archetype = np.argmax(arch_corrs)

    return get_genes_from_spatial_archetype(sdge, gene_names, archetypes, archetype,
                                            pval_threshold=pval_threshold)


######################
# plotting functions #
######################

def plot_gene_pattern(locations, sdge, gene, folder, gene_names, num_cells):
    plt.figure(figsize=(2, 1))
    plt.scatter(locations[:, 0], locations[:, 1], c=sdge[np.argwhere(gene_names == gene), :].flatten(), s=70,
                          cmap="BuPu" # for zebrafish coloring
    )
    plt.axis('off')
    # plt.title(gene)
    plt.savefig(folder + str(num_cells) + '_' + gene + '.png')
    plt.clf()

    
def plot_gene_patterns(locations, sdge, genes, folder, gene_names, num_cells):
    """genes are given as a list: ['gene1', 'gene2']"""
    num_rows = int(round(np.sqrt(len(genes))))
    xf = len(np.unique(locations[:, 0]))
    yf = len(np.unique(locations[:, 1]))
    plt.figure(figsize=(12, 8)) # (8, 4) for zebrafish, (18, 10) for half bdtnp genes # (12, 8) for cell cycle
    idx = 1
    for gene in genes:
        plt.subplot(num_rows, np.ceil(len(genes)/num_rows), idx)
        plt.scatter(locations[:, 0], locations[:, 1], c=sdge[np.argwhere(gene_names == gene), :].flatten(),
                  # cmap="BuPu" # for zebrafish coloring
        )
        plt.title(gene)
        plt.axis('off')
        idx += 1
    plt.tight_layout()
    plt.savefig(folder + str(num_cells) + '_cells_'
                + str(locations.shape[0]) + '_locations' + '.png')
    plt.clf()

def plot_gene_patterns_1D(locations, sdge, genes, folder, gene_names, num_cells):
    num_rows = int(round(np.sqrt(len(genes))))
    xf = len(np.unique(locations))
    plt.figure(figsize=(16, 12))
    idx = 1
    for gene in genes:
        plt.subplot(num_rows, np.ceil(len(genes)/num_rows), idx)
        plt.scatter(locations, sdge[np.argwhere(gene_names == gene), :].flatten())
        plt.title(gene)
        plt.axis('off')
        idx += 1
    plt.tight_layout()
    plt.savefig(folder + str(num_cells) + '_cells_'
                + str(locations.shape[0]) + '_locations' + '.png')
    plt.clf()

    
def plot_dendgrogram(sdge, folder):
    """Plots the dendrogram of the hierarchical clustering to inspect and choose
    the number of clusters / archetypes."""
    plt.figure(figsize=(25, 10))
    hierarchy.dendrogram(hierarchy.ward(sdge), leaf_rotation=90.)
    plt.savefig(folder + 'dendrogram.png')


def plot_archetypes(locations, archetypes, clusters, gene_corrs, gene_set, folder):
    """Plots the spatial archetypes onto a file.
    locations -- the grid / target space
    archetypes -- the spatial archetypes
    clusters   -- the clusters
    gene_corrs -- the gene correlations
    gene_set   -- the genes that were used to find the archetypes """
    print ('Plotting gene archetypes ... ', flush=True, end='')
    num_rows = int(round(np.sqrt(max(clusters))))
    plt.figure(figsize=(num_rows*2.5*2, num_rows*2.5))
    idx = 1
    for archetype in range(1, max(clusters)+1):
        which_genes = np.where(clusters == archetype)[0]
        plt.subplot(num_rows, np.ceil(max(clusters)/num_rows), idx)
        plt.scatter(locations[:, 0], locations[:, 1],
                    c=archetypes[archetype-1, : ])
        plt.title('archetype ' + str(archetype) + '\n' +
                  '\n'.join(wrap(', '.join(gene_set[which_genes][np.argsort(gene_corrs[which_genes])[-5:]]), 40)))
        idx += 1
        plt.tight_layout()
        plt.savefig(folder + 'spatial_archetypes.png')
    print ('done')


########
# main #
########
    
if __name__ == '__main__':
    pass
    
