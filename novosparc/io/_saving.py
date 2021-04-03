import time
import numpy as np
import os
import novosparc

def write_sdge_to_disk(tissue, folder):
    sdge = tissue.sdge
    num_cells = tissue.num_cells
    num_locations = tissue.num_locations

    np.savetxt(os.path.join(folder, 'sdge_' + str(num_cells) + '_cells_'
                            + str(num_locations) + '_locations.txt'), sdge, fmt='%.4e')

def save_gene_pattern_plots(tissue, gene_list_to_plot,  folder):
    novosparc.pl.plot_gene_patterns(tissue.locations, tissue.sdge, gene_list_to_plot,
                                    folder=folder,
                                    gene_names=tissue.gene_names, num_cells=tissue.num_cells)

def save_spatially_informative_gene_pattern_plots(tissue, gene_count_to_plot,  folder):
    novosparc.pl.plot_gene_patterns(tissue.locations, tissue.sdge, tissue.spatially_informative_genes[:gene_count_to_plot]['genes'],
                                    folder=folder,
                                    gene_names=tissue.gene_names, num_cells=tissue.num_cells,
                                    prefix='_spatially_important_')