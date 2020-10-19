import novosparc
import time
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr

if __name__ == '__main__':


    ######################################
    # 1. Set the data and output paths ###
    ######################################

    dataset_path = '/pathto/dge.txt' # this could be the dge file, or also can be a 10x mtx folder
    target_space_path = '/pathto/locations.txt' # location coordinates if exist
    output_folder = '/pathto/output' # folder to save the results, plots etc.

    #######################################
    # 2. Read the dataset and subsample ###
    #######################################

    # Read the dge. this assumes the file formatted in a way that genes are columns and cells are rows.
    # If the data is the other way around, transpose the dataset object (e.g dataset=dataset.T)
    dataset = novosparc.io.load_data(dataset_path)

    # Optional: downsample number of cells.
    cells_selected, dataset = novosparc.pp.subsample_dataset(dataset, min_num_cells=500, max_num_cells=1000)
    
    # Optional: Subset to the highly variable genes
    dataset.raw = dataset # this stores the current dataset with all the genes for future use
    hvg_path = '/pathto/hvg.txt'

    # a file for a list of highly variable genes can be provided. or directly a gene list provided 
    # with the argument 'gene_list'. The whole process can be done also with scanpy
    dataset, hvg = novosparc.pp.subset_to_hvg(dataset, hvg_file = hvg_path) 

    # Load the location coordinates from file if it exists
    locations = novosparc.io.load_target_space(target_space_path, cells_selected, coords_cols=['xcoord', 'ycoord'])

    # Alternatively, construct a square target grid
    locations = novosparc.rc.construct_target_grid(num_cells)

    #########################################
    # 3. Setup and spatial reconstruction ###
    #########################################

    tissue = novosparc.cm.Tissue(dataset=dataset, locations=locations, output_folder=output_folder) # create a tissue object
    tissue.setup_reconstruction(num_neighbors_s = 5, num_neighbors_t = 5) 

    # Optional: use marker genes
    tissue.setup_reconstruction(markers_to_use=markers_to_use, atlas_matrix=atlas_matrix)

    # alpha parameter controls the reconstruction. Set 0 for de novo, between
    # 0 and 1 in case markers are available.
    tissue.reconstruct(alpha_linear=0) # reconstruct with the given alpha value

    # calculate spatially informative genes after reconstruction
    tissue.calculate_spatially_informative_genes() 


    #############################################
    # 4. Save the results and plot some genes ###
    #############################################

    # save the sdge to file
    novosparc.io.write_sdge_to_disk(tissue, output_folder)

    # plot some genes and save them
    gene_list_to_plot = ['gene1', 'gene2', 'gene3', 'gene4']
    novosparc.io.save_gene_pattern_plots(tissue=tissue, gene_list_to_plot=gene_list_to_plot, folder=output_folder)
    novosparc.io.save_spatially_informative_gene_pattern_plots(tissue=tissue, gene_count_to_plot=10, folder=output_folder)
