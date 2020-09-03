import novosparc
import numpy as np
import scanpy as sc
import os

if __name__ == '__main__':

    ######################################
    # 1. Set the data and output paths ###
    ######################################
    dataset_path = 'novosparc/datasets/intestine/dge.tsv.gz'
    target_space_path = 'novosparc/datasets/intestine/zones.tsv'
    dirname = os.path.dirname(__file__)
    output_folder = os.path.join(dirname, 'output_intestine')

    #######################################
    # 2. Read the dataset and normalize ###
    #######################################
    dataset = novosparc.io.load_data(dataset_path).T
    sc.pp.normalize_total(dataset, target_sum=1, inplace=True)

    # Read the annnotated spatial information
    locations_original = np.loadtxt(target_space_path, skiprows=1, usecols=range(1, 4))
    locations_original = locations_original[:, 2]
    grid_len = len(np.unique(locations_original))
    locations = np.vstack((range(grid_len), np.ones(grid_len))).T

    # Optional: Subsample the cells
    # num_cells = len(dataset.obs)
    # cells_selected, dataset = novosparc.pp.subsample_dataset(dataset, num_cells-1, num_cells)
    # locations_original = locations_original[cells_selected]

    dge_full = dataset.X
    # Compute mean dge over original zones 
    dge_full_mean = np.zeros((grid_len, dge_full.shape[1]))
    for i in range(grid_len):
        indices = np.argwhere(locations_original == i).flatten()
        temp = np.mean(dge_full[indices, :], axis=0)
        dge_full_mean[i, :] = temp
    dge_full_mean = dge_full_mean.T

    gene_names = np.array(dataset.var.index.tolist())
    # Select variable genes
    var_genes = np.argsort(np.divide(np.var(dge_full.T, axis=1), np.mean(dge_full.T, axis=1) + 0.0001))
    dge = dge_full[:, var_genes[-1000:]]

    #########################################
    # 3. Setup and reconstruct the tissue ###
    #########################################

    tissue = novosparc.cm.Tissue(dataset, locations, output_folder)
    tissue.setup_reconstruction(num_neighbors_t=2)
    tissue.reconstruct(alpha_linear=0)  # alpha is 0 for de novo reconstruction

    # Compute mean expression distribution over embedded zones 
    mean_exp_new_dist = np.zeros((grid_len, grid_len))
    for i in range(grid_len):
        indices = np.argwhere(locations_original == i).flatten()
        temp = np.sum(tissue.gw[indices, :], axis=0)
        mean_exp_new_dist[i, :] = temp / np.sum(temp)

    #########################################
    # 4. Write data to disk for further use #
    #########################################

    novosparc.io.write_sdge_to_disk(tissue, output_folder)

    ###########################################################################################
    # 5. Plot histogram showing the distribution over embedded zones for each original zone #
    ###########################################################################################

    novosparc.pl.plot_histogram_intestine(mean_exp_new_dist, folder=output_folder)

    ###########################################################################################
    # 6. Plot spatial expression of a few gene groups for the original and embedded zones #
    ###########################################################################################

    novosparc.pl.plot_spatial_expression_intestine(dge_full_mean, tissue.sdge, gene_names, folder=output_folder)