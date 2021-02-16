from __future__ import print_function

###########
# imports #
###########

import numpy as np
# from sklearn import manifold, datasets # not used, remove in next version
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import cdist
from scipy.cluster import hierarchy
from scipy.stats import pearsonr
import ot
import time
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
import os

#############
# functions #
#############

def setup_for_OT_reconstruction(dge, locations, num_neighbors_source = 5, num_neighbors_target = 5,
                                locations_metric='minkowski', locations_metric_p=2,
                                expression_metric='minkowski', expression_metric_p=2, verbose=True):
    start_time = time.time()
    if verbose:
        print ('Setting up for reconstruction ... ', end='', flush=True)

    # Shortest paths matrices at target and source spaces
    num_neighbors_target = num_neighbors_target # number of neighbors for nearest neighbors graph at target
    A_locations = kneighbors_graph(locations, num_neighbors_target, mode='connectivity', include_self=True,
                                   metric=locations_metric, p=locations_metric_p)
    sp_locations = dijkstra(csgraph=csr_matrix(A_locations), directed=False, return_predecessors=False)
    sp_locations_max = np.nanmax(sp_locations[sp_locations != np.inf])
    sp_locations[sp_locations > sp_locations_max] = sp_locations_max #set threshold for shortest paths

    num_neighbors_source = num_neighbors_source # number of neighbors for nearest neighbors graph at source
    A_expression = kneighbors_graph(dge, num_neighbors_source, mode='connectivity', include_self=True,
                                    metric=expression_metric, p=expression_metric_p)
    sp_expression = dijkstra(csgraph=csr_matrix(A_expression), directed=False, return_predecessors=False)
    sp_expression_max = np.nanmax(sp_expression[sp_expression != np.inf])
    sp_expression[sp_expression > sp_expression_max] = sp_expression_max #set threshold for shortest paths

    # Set normalized cost matrices based on shortest paths matrices at target and source spaces
    cost_locations = sp_locations / sp_locations.max()
    cost_locations -= np.mean(cost_locations)
    cost_expression = sp_expression / sp_expression.max()
    cost_expression -= np.mean(cost_expression)

    if verbose:
        print('done (', round(time.time()-start_time, 2), 'seconds )')
    return cost_expression, cost_locations

    
def create_space_distributions(num_locations, num_cells):
    """Creates uniform distributions at the target and source spaces.
    num_locations -- the number of locations at the target space
    num_cells     -- the number of single-cells in the data."""
    p_locations = ot.unif(num_locations)
    p_expression = ot.unif(num_cells)
    return p_locations, p_expression


def write_sdge_to_disk(sdge, num_cells, num_locations, folder):
    """Writes the spatial gene expression matrix to disk for further usage.
    sdge      -- the sdge
    num_cells -- the number of single-cells in the data
    num_locations -- the number of locations at the target space
    folder    -- the folder to output the file"""
    start_time = time.time()
    print ('Writing data to disk ...', end='', flush=True)
    np.savetxt(os.path.join(folder, 'sdge_' + str(num_cells) + '_cells_'
               + str(num_locations) + '_locations.txt'), sdge, fmt='%.4e')
    print ('done (', round(time.time()-start_time, 2), 'seconds )')
    
    
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
    
