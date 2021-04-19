from __future__ import print_function

###########
# imports #
###########

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.cluster import hierarchy
from textwrap import wrap
import novosparc
import os
import copy
import pandas as pd
from scipy.spatial.distance import squareform, pdist
#############
# functions #
#############

def plot_mapped_cells(locations, gw, cells, folder, 
                      size_x=16, size_y=12, pt_size=20, cmap='viridis'):
    """Plots the mapped locations of a cell population.
    locations -- the locations of the target space
    gw        -- the Gromow-Wasserstein matrix computed during the reconstruction
    cells     -- the queried cellls as a numpy array
    folder    -- the folder to save the .png output.
    pt_size   -- the size of the points
    cmap      -- custom colormap. Only used for 2D reconstructions
    """
    plt.figure(figsize=(size_x, size_y))
    if locations.shape[1] == 1:
        plt.scatter(locations, np.sum(gw[cells, :], axis=0), s=pt_size)
    if locations.shape[1] == 2:
        plt.scatter(locations[:, 0], locations[:, 1],
            c=np.sum(gw[cells, :], axis=0), s=pt_size, cmap=cmap)
    plt.savefig(os.path.join(folder, 'mapped_cells.png'))
    plt.close()


def plot_gene_patterns(locations, sdge, genes, folder, gene_names, num_cells,
                       size_x=16, size_y=12, pt_size=20, tit_size=15, cmap='viridis', prefix=''):
    """Plots gene expression patterns on the target space.
    locations  -- the locations of the target space
    sdge       -- the sdge computed from the reconstruction
    genes      -- the genes to plot as a list: ['gene1', 'geme2', ...]
    folder     -- the folder to save the .png output.
    gene_names -- an numpy array of all genes appearing in the sdge
    num_cells  -- the number of cells used for the reconstruction
    size_x     -- the width of the resulting figure
    size_y     -- the height of the resulting figure
    pt_size    -- the size of the points
    cmap       -- custom colormap. Only used for 2D reconstructions
    """
    num_rows = int(round(np.sqrt(len(genes))))
    plt.figure(figsize=(size_x, size_y))
    
    idx = 1
    for gene in genes:
        plt.subplot(num_rows, np.ceil(len(genes)/num_rows), idx)
        if locations.shape[1] == 1:
            plt.scatter(locations, sdge[np.argwhere(gene_names == gene), :].flatten(),
                        s=pt_size)
        if locations.shape[1] == 2:
            plt.scatter(locations[:, 0], locations[:, 1], 
                        c=sdge[np.argwhere(gene_names == gene), :].flatten(),
                        s=pt_size, cmap=cmap)
        plt.title(gene, size=tit_size)
        plt.axis('off')
        idx += 1
            
    plt.tight_layout()
    if os.path.isdir(folder):
        plt.savefig(os.path.join(folder, str(num_cells) + '_cells_'
            + str(locations.shape[0]) + '_locations' + prefix + '.png'))
        plt.close()
    else:
        plt.show()


def embedding(dataset, color, title=None, size_x=None, size_y=None,
                          pt_size=10, tit_size=15, dpi=100):
    """
    Plots fields (color) of Scanpy AnnData object on spatial coordinates
    dataset -- Scanpy AnnData with 'spatial' matrix in obsm containing the spatial coordinates of the tissue
    color -- a list of fields - gene names or columns from obs to use for color
    """
    title = color if title is None else title
    ncolor = len(color)
    per_row = 3
    per_row = ncolor if ncolor < per_row else per_row
    nrows = int(np.ceil(ncolor / per_row))
    size_x = 5 * per_row if size_x is None else size_x
    size_y = 3 * nrows if size_y is None else size_y
    fig, axs = plt.subplots(nrows, per_row, figsize=(size_x, size_y), dpi=dpi)
    xy = dataset.obsm['spatial']
    x = xy[:, 0]
    y = xy[:, 1] if xy.shape[1] > 1 else np.ones_like(x)
    axs = axs.flatten() if type(axs) == np.ndarray else [axs]
    for ax in axs:
        ax.axis('off')

    for i, g in enumerate(color):
        if g in dataset.var_names:
            values = dataset[:, g].X
        elif g in dataset.obs.columns:
            values = dataset.obs[g]
        else:
            continue
        axs[i].scatter(x, y, c=np.array(values), s=pt_size)
        axs[i].set_title(title[i], size=tit_size)

    plt.show()
    plt.tight_layout()

def plot_histogram_intestine(mean_exp_new_dist, folder):
    plt.figure(figsize=(5, 5))
    ax = plt.gca()
    im = ax.imshow(mean_exp_new_dist.T,origin='lower')
    my_xticks = ['crypt','V1','V2','V3','V4','V5','V6']
    x = range(mean_exp_new_dist.shape[0])
    plt.xticks(x, my_xticks)
    my_yticks = ['0','1','2','3','4','5','6']
    my_yticks.reverse()
    plt.yticks(range(mean_exp_new_dist.shape[0]), my_yticks)
    plt.ylabel('Embedded value')
    plt.xlabel('Villus zone')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    im.set_clim(0, 1)
    plt.savefig(os.path.join(folder, 'histogram_intestine.png'))
    plt.clf()

def plot_spatial_expression_intestine(dge_full_mean, sdge, gene_names, folder):
    
    gene_list = ['Apobec1', 'Apob', 'Apoa4', 'Apoa1', 'Npc1l1', 'Slc15a1', 'Slc5a1', 
                 'Slc2a5', 'Slc2a2', 'Slc7a9', 'Slc7a8', 'Slc7a7']
    
    zonated_lst=[]
    for gene in gene_list:
        zonated_lst = np.append(zonated_lst, np.argwhere(gene_names == gene))
    zonated_lst = zonated_lst.astype(int)

    plt.subplots(2,1, sharex=True, figsize=(7,5.5))
    
    plt.subplot(2,1,1)    
    x = range(7)
    y = dge_full_mean[zonated_lst,:].T
    y_AC = np.mean(y[:,0:5],axis=1)
    y_P = y[:,5]
    y_C = np.mean(y[:,6:9],axis=1)
    y_AA = np.mean(y[:,9:],axis=1)
    y = np.vstack((y_AA/y_AA.max(),y_C/y_C.max(),y_P/y_P.max(),y_AC/y_AC.max()))
    ax = plt.gca()
    im = ax.imshow(y)
    my_xticks = ['crypt','V1','V2','V3','V4','V5','V6']
    plt.xticks(x, my_xticks)
    plt.xlabel('Villus zones')
    my_yticks = ['Amino acids','Carbohydrates','Peptides', 'Apolipoproteins' '\n' 'Cholesterol']
    plt.yticks(range(len(my_yticks)), my_yticks)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    plt.subplot(2,1,2)    
    x = range(7)
    y = sdge[zonated_lst,:].T
    y_AC = np.mean(y[:,0:5],axis=1)
    y_P = y[:,5]
    y_C = np.mean(y[:,6:9],axis=1)
    y_AA = np.mean(y[:,9:],axis=1)
    y = np.vstack((y_AA/y_AA.max(),y_C/y_C.max(),y_P/y_P.max(),y_AC/y_AC.max()))
    ax = plt.gca()
    im = ax.imshow(y)
    my_xticks = ['0','1','2','3','4','5','6']
    plt.xticks(x, my_xticks)
    plt.xlabel('Embedded zones')
    my_yticks = ['Amino acids','Carbohydrates','Peptides',r'Apolipoproteins' '\n' 'Cholesterol']
    plt.yticks(range(len(my_yticks)), my_yticks)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.tight_layout()

    plt.savefig(os.path.join(folder, 'spatial_expression_intestine.png'))
    plt.clf()

def plot_dendrogram(sdge, folder, size_x=25, size_y=10):
    """Plots the dendrogram of the hierarchical clustering to inspect and choose
    the number of clusters / archetypes.
    """
    plt.figure(figsize=(size_x, size_y))
    hierarchy.dendrogram(hierarchy.ward(sdge), leaf_rotation=90.)
    plt.savefig(os.path.join(folder, 'dendrogram.png'))


def plot_archetypes(locations, archetypes, clusters, gene_corrs, gene_set, folder):
    """Plots the spatial archetypes onto a file.
    locations  -- the locations of the target space
    archetypes -- the spatial archetypes
    clusters   -- the clusters
    gene_corrs -- the gene correlations
    gene_set   -- the genes that were used to find the archetypes
    """
    
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
        plt.savefig(os.path.join(folder, 'spatial_archetypes.png'))
    plt.close()


def plot_transport_entropy_dist(tissue, tit_size=20, fonts=16, fonts_ticks=20):
    """
    Plots the distribution of entropy of locations transport values for each cell.
    tissue -- novoSpaRc Tissue object with the transport to evaluate
    Displays histograms for:
        - novoSpaRc - the given mapping
        - novoSpaRc with shuffled atlas - if an atlas is used, shuffles each gene independently over all locations, and recomputes ot with the same params
        - Random - projection of a random matrix onto the coupling space (set marginal distributions)
        - Outer product - entropy for the naive outer product coupling (uniform if marginals are uniform)
    """

    num_cells, num_locations = tissue.gw.shape

    p = tissue.p_expression
    q = tissue.p_locations
    epsilon = tissue.epsilon

    # compute uniform coupling
    unif_coupling = np.outer(p, q)

    # compute random coupling
    rand_coupling = novosparc.analysis.compute_random_coupling(p, q, epsilon)

    # if there is an atlas, compute reconstruction with shuffled atlas
    has_atlas = tissue.atlas_matrix is not None
    if has_atlas:
        tissue_shuf = copy.deepcopy(tissue)
        atlas_matrix_shuf = tissue_shuf.atlas_matrix

        locs_shuf = np.arange(num_locations)

        for g in np.arange(atlas_matrix_shuf.shape[1]):
            np.random.shuffle(locs_shuf)
            atlas_matrix_shuf[:, g] = atlas_matrix_shuf[locs_shuf, g]

        tissue_shuf.setup_linear_cost(markers_to_use=tissue_shuf.markers_to_use, atlas_matrix=atlas_matrix_shuf)
        tissue_shuf.reconstruct(alpha_linear=tissue_shuf.alpha_linear, epsilon=tissue_shuf.epsilon)

    # compute entropies
    ent_T = novosparc.an.get_cell_entropy(tissue.gw)
    ent_T_unif = novosparc.an.get_cell_entropy(unif_coupling)
    ent_T_rproj = novosparc.an.get_cell_entropy(rand_coupling)
    ent_T_shuf = novosparc.an.get_cell_entropy(tissue_shuf.gw) if has_atlas else None

    # plot entropy distributions
    min_ent = np.min(ent_T)
    max_ent = np.min(ent_T_unif) * 1.1
    bins = np.linspace(min_ent, max_ent, 40)
    kwargs = dict(histtype='stepfilled', alpha=0.3, bins=bins)
    fig = plt.figure(figsize=[8, 6])
    ax = fig.add_subplot(111)
    ax.hist(ent_T, label='novoSpaRc', **kwargs)
    ax.hist(ent_T_rproj, label='Random', **kwargs)
    ax.hist(ent_T_unif, label='Outer product', **kwargs)
    ax.hist(ent_T_shuf, label='novoSpaRc with shuffled atlas ', **kwargs)
    ax.set_title('Entropy distribution of transport matrices', size=tit_size)
    ax.set_xlabel('Entropy', size=tit_size)
    lg = ax.legend(fontsize=fonts, loc='upper left', title='Transport matrix')
    lg.get_title().set_fontsize(fonts)
    ax.tick_params(labelsize=fonts_ticks)
    plt.show()

    return ent_T, ent_T_unif, ent_T_rproj, ent_T_shuf


def plot_morans_dists(genes_with_scores, gene_groups, tit_size=20, fonts=16, fonts_ticks=20):
    """
    Overlay Moran's I distributions of given gene groups
    genes_with_scores -- pandas DataFrame with fields ['genes', 'mI'], for the gene and its corresponding Moran's I value
    gene_groups       -- dictionary of gene groups to show, key - group name, value - list of gene names
    """
    min_mI = 0.8
    max_mI = 0.7
    genes_with_scores.index = genes_with_scores['genes']
    for gg_desc, gg in gene_groups.items():
        mIs = genes_with_scores.loc[gg]['mI']
        min_mI = min(mIs) if min(mIs) < min_mI else min_mI
        max_mI = max(mIs) if max(mIs) > max_mI else max_mI

    min_mI = np.floor(10 * min_mI) / 10
    max_mI = np.ceil(10 * max_mI) / 10

    bins = np.linspace(min_mI, max_mI, 40)
    kwargs = dict(histtype='stepfilled', alpha=0.3, bins=bins)
    fig = plt.figure(figsize=[8, 6])
    ax = fig.add_subplot(111)
    for gg_desc, gg in gene_groups.items():
        ax.hist(genes_with_scores.loc[gg]['mI'], label=gg_desc, **kwargs)
    ax.set_title('Moran`s I for %s genes' % ', '.join(list(gene_groups.keys())), size=tit_size)
    ax.set_xlabel('Moran`s I', size=tit_size)
    lg = ax.legend(fontsize=fonts, loc='upper left', title='Gene group')
    lg.get_title().set_fontsize(fonts)
    ax.tick_params(labelsize=fonts_ticks)
    plt.show()

def plot_exp_loc_dists(exp, locations, tit_size=15, nbins=10):
    """
    Plots expression distances vs physical distances over locations
    exp       -- spatial expression over locations (locations x genes)
    locations -- spatial coordinates of locations (locations x dimensions)
    """
    locs_exp_dist = squareform(pdist(exp))
    locs_phys_dist = squareform(pdist(locations))
    exp_col = 'Locations expression distance'
    phys_col = 'Locations physical distance'
    phys_col_bin = 'Locations physical distance bin'

    df = pd.DataFrame({exp_col: locs_exp_dist.flatten(),
                      phys_col: locs_phys_dist.flatten()})

    lower, higher = int(df[phys_col].min()), int(np.ceil(df[phys_col].max()))
    edges = range(lower, higher, int((higher - lower)/nbins)) # the number of edges is 8
    lbs = ['(%d, %d]'%(edges[i], edges[i+1]) for i in range(len(edges)-1)]
    df[phys_col_bin] = pd.cut(df[phys_col], bins=nbins, labels=lbs, include_lowest=True)

    df.boxplot(column=[exp_col], by=[phys_col_bin], grid=False, fontsize=tit_size)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel(exp_col, size=tit_size)
    plt.xlabel(phys_col, size=tit_size)
    plt.title('')
    plt.show()
