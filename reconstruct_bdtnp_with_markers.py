###########
# imports #
###########

import novosparc
import time
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr

if __name__ == '__main__':

    ###################################
    # 1. Import and subset the data ###
    ###################################
    start_time = time.time()
    print ('Loading data ... ', end='', flush=True)

    # Read the BDTNP database
    gene_names = np.genfromtxt('novosparc/datasets/bdtnp/dge.txt', usecols=range(84),
                          dtype='str', max_rows=1)
    dge = np.loadtxt('novosparc/datasets/bdtnp/dge.txt', usecols=range(84), skiprows=1)

    # Optional: downsample number of cells
    cells_selected, dge = novosparc.pp.subsample_dge(dge, 2000, 2500)
    num_cells = dge.shape[0]
    
    # Choose a number of markers to use for reconstruction
    num_markers = int(np.random.randint(1, 5, 1))
    markers_to_use = np.random.choice(dge.shape[1], num_markers, replace=False)

    print ('done (', round(time.time()-start_time, 2), 'seconds )')
    
    ################################
    # 2. Set the target space grid #
    ################################

    print ('Reading the target space ... ', end='', flush=True)    
    # Read and use the bdtnp geometry
    locations = np.loadtxt('novosparc/datasets/bdtnp/geometry.txt', usecols=range(3), skiprows=1)
    locations = locations[:, [0, 2]]
    locations = locations[cells_selected, :] # downsample to the cells selected above
    num_locations = locations.shape[0]
    print ('done')

    ######################################
    # 3. Setup for the OT reconstruction #
    ######################################
    
    cost_expression, cost_locations = novosparc.rc.setup_for_OT_reconstruction(dge[:, np.setdiff1d(np.arange(dge.shape[1]),
                                                                                                   markers_to_use)],
                                                                               locations,
                                                                               num_neighbors_source = 5,
                                                                               num_neighbors_target = 5)

    cost_marker_genes = cdist(dge[:, markers_to_use]/np.amax(dge[:, markers_to_use]),
                              dge[:, markers_to_use]/np.amax(dge[:, markers_to_use]))

    #############################
    # 4. Spatial reconstruction #
    #############################

    start_time = time.time()
    print ('Reconstructing spatial information with', num_markers,
           'markers:', num_cells, 'cells and',
           locations.shape[0], 'locations ... ')
    
    # Distributions at target and source spaces
    p_locations, p_expression = novosparc.rc.create_space_distributions(num_locations, num_cells)

    alpha_linear = 0.5
    gw = novosparc.rc._GWadjusted.gromov_wasserstein_adjusted_norm(cost_marker_genes, cost_expression, cost_locations,
                                              alpha_linear, p_expression, p_locations,
                                              'square_loss', epsilon=5e-4, verbose=True)
    sdge = np.dot(dge.T, gw)
    
    print (' ... done (', round(time.time()-start_time, 2), 'seconds )')

    #########################################
    # 5. Write data to disk for further use #
    #########################################

    novosparc.rc.write_sdge_to_disk(sdge, num_cells, num_locations, 'output_bdtnp')
    ###########################
    # 6. Plot gene expression #
    ###########################

    gene_list_to_plot = ['ftz', 'Kr', 'sna', 'zen2']
    novosparc.pl.plot_gene_patterns(locations, sdge, gene_list_to_plot,
                                    folder='output_bdtnp/',
                                    gene_names=gene_names, num_cells=num_cells)

    ###################################
    # 7. Correlate results with BDTNP #
    ###################################
    
    with open('output_bdtnp/results.txt', 'a') as f:
        f.write('number_cells,,number_markers,' +  ','.join(gene_names) + '\n')
        f.write(str(num_cells) + ',' + str(num_markers) + ',')
        for i in range(len(gene_names)):
            f.write(str(round(pearsonr(sdge[i, :], dge[:, i])[0], 2)) + ',')

    ############################################
    # 8. Calculate spatially informative genes #
    ############################################
    important_gene_names = novosparc.analysis.morans(sdge, gene_names, locations)
    novosparc.pl.plot_gene_patterns(locations, sdge, important_gene_names,
                                    folder='output_bdtnp/',
                                    gene_names=gene_names, num_cells=num_cells, prefix='_spatially_important_')














    


