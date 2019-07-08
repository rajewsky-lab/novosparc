from __future__ import print_function

###########
# imports #
###########

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

#############
# functions #
#############

def plot_mapped_cells(locations, gw, cells, folder, 
                      size_x=16, size_y=12, pt_size=20, cmap='viridis'):
    """Plots the mapped locations of a cell population.

    Keyword arguments:
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
    plt.savefig(folder.replace('/', '') + '/' + 'mapped_cells.png')
    plt.close()


def plot_gene_patterns(locations, sdge, genes, folder, gene_names, num_cells,
                       size_x=16, size_y=12, pt_size=20, cmap='viridis'):
    """Plots gene expression patterns on the target space.

    Keyword arguments:
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
        plt.title(gene)
        plt.axis('off')
        idx += 1
            
    plt.tight_layout()
    plt.savefig(folder.replace('/', '') + '/' 
        + str(num_cells) + '_cells_'
        + str(locations.shape[0]) + '_locations' + '.png')
    plt.close()
    

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
    plt.savefig(folder + 'histogram_intestine' + '.png')
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

    plt.savefig(folder.replace('/', '') + 'spatial_expression_intestine' + '.png')
    plt.clf()

def plot_dendrogram(sdge, folder, size_x=25, size_y=10):
    """Plots the dendrogram of the hierarchical clustering to inspect and choose
    the number of clusters / archetypes.
    """
    plt.figure(figsize=(size_x, size_y))
    hierarchy.dendrogram(hierarchy.ward(sdge), leaf_rotation=90.)
    plt.savefig(folder.replace('/', '') + '/dendrogram.png')


def plot_archetypes(locations, archetypes, clusters, gene_corrs, gene_set, folder):
    """Plots the spatial archetypes onto a file.
    
    Keyword arguments:
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
        plt.savefig(folder.replace('/', '') + '/spatial_archetypes.png')
    plt.close()
