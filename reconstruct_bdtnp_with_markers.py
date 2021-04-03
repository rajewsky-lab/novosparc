import novosparc
import time
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
import os
import scanpy as sc

if __name__ == '__main__':

    ######################################
    # 1. Set the data and output paths ###
    ######################################

    dataset_path = 'novosparc/datasets/bdtnp/dge.txt'
    target_space_path = 'novosparc/datasets/bdtnp/geometry.txt'
    dirname = os.path.dirname(__file__)
    output_folder = os.path.join(dirname, 'output_bdtnp')

    #############################################
    # 2. Read the dataset and subsample cells ###
    #############################################

    # Read the dge as an anndata object.
    # Refer to https://anndata.readthedocs.io/en/latest/ for details of the datatype
    dataset = novosparc.io.load_data(dataset_path)

    # Subsample the cells
    cells_selected, dataset = novosparc.pp.subsample_dataset(dataset, 500, 1000)

    # Load the location coordinates from file
    locations = novosparc.io.load_target_space(target_space_path, cells_selected, coords_cols=['xcoord', 'zcoord'], sep=' ')

    # Choose a number of markers to use for reconstruction
    num_markers = int(np.random.randint(1, 5, 1))
    markers_to_use = np.random.choice(len(dataset.var), num_markers, replace=False)
    atlas_matrix = dataset.X[:, markers_to_use]

    #########################################
    # 3. Setup and spatial reconstruction ###
    #########################################

    tissue = novosparc.cm.Tissue(dataset=dataset, locations=locations, output_folder=output_folder) # create a tissue object
    tissue.setup_reconstruction(markers_to_use=markers_to_use, atlas_matrix=atlas_matrix) # setup construction (optional: using markers)
    tissue.reconstruct(alpha_linear=0.5) # reconstruct with the given alpha value

    tissue.calculate_spatially_informative_genes() # calculate spatially informative genes

    #############################################
    # 4. Save the results and plot some genes ###
    #############################################

    # save the sdge to file
    novosparc.io.write_sdge_to_disk(tissue, output_folder)

    # plot some genes and save them
    gene_list_to_plot = ['ftz', 'Kr', 'sna', 'zen2']
    novosparc.io.save_gene_pattern_plots(tissue=tissue, gene_list_to_plot=gene_list_to_plot, folder=output_folder)
    novosparc.io.save_spatially_informative_gene_pattern_plots(tissue=tissue, gene_count_to_plot=10, folder=output_folder)

    ###################################
    # 5. Correlate results with BDTNP #
    ###################################

    with open(os.path.join(output_folder, 'results.txt'), 'a') as f:
        f.write('number_cells,,number_markers,' + ','.join(tissue.gene_names) + '\n')
        f.write(str(tissue.num_cells) + ',' + str(num_markers) + ',')
        for i in range(len(tissue.gene_names)):
            f.write(str(round(pearsonr(tissue.sdge[i, :], tissue.dge[:, i])[0], 2)) + ',')
