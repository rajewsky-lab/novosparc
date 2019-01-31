#########
# about #
#########

__version__ = "0.1.1"
__author__ = ["Nikos Karaiskos", "Mor Nitzan"]
__status__ = "beta"
__licence__ = "GPL"
__email__ = ["nikolaos.karaiskos@mdc-berlin.de", "mornitz@gmail.com"]

###########
# imports #
###########

from novosparc import *

if __name__ == '__main__':

    ###################################
    # 1. Import and subset the data ###
    ###################################
    start_time = time.time()
    print ('Loading data ... ', end='', flush=True)

    # Read the BDTNP database
    zfile = zipfile.ZipFile('datasets/intestine/dge.tsv.zip')
    zfile.extract('dge.tsv')
    gene_names = np.genfromtxt('dge.tsv', usecols=0, dtype='str', skip_header=1)
    dge = np.loadtxt('dge.tsv',skiprows=1,usecols=range(1,1384))
    dge = dge.T
    
    # Optional: downsample number of cells
    num_cells = dge.shape[0] # all cells are used
    cells_selected = np.random.choice(dge.shape[0], num_cells, replace=False)
    dge = dge[cells_selected, :]    
    
    # Select variable genes
    var_genes = identify_highly_variable_genes(dge, low_x=1, high_x=8, low_y=1, do_plot=False)
    dge = dge[:, var_genes]    
    
    print ('done (', round(time.time()-start_time, 2), 'seconds )')
    
    ################################
    # 2. Set the target space grid #
    ################################

    print ('Reading the target space ... ', end='', flush=True)    
    # Read and use a 1d grid
    
    locations = np.vstack((range(7),np.ones(7))).T
    
    print ('done')

    ######################################
    # 3. Setup for the OT reconstruction #
    ######################################
    
    cost_expression, cost_locations = setup_for_OT_reconstruction_1d(dge, locations, 
                                                                     num_neighbors_source = 5, num_neighbors_target = 2)
    
    # no marker genes are used
    cost_marker_genes = np.ones((num_cells,len(locations)))

    #############################
    # 4. Spatial reconstruction #
    #############################

    start_time = time.time()
    print ('Reconstructing spatial information with', num_cells, 'cells and',
           locations.shape[0], 'locations ... ')
    
    # Distributions at target and source spaces
    p_locations = ot.unif(len(locations))
    p_expression = ot.unif(num_cells)

    alpha_linear = 0
    gw = gwa.gromov_wasserstein_adjusted_norm(cost_marker_genes, cost_expression, cost_locations,
                                              alpha_linear, p_expression, p_locations,
                                              'square_loss', epsilon=5e-4, verbose=True)
    sdge = np.dot(dge.T, gw)
    
    print (' ... done (', round(time.time()-start_time, 2), 'seconds )')

    #########################################
    # 5. Write data to disk for further use #
    #########################################

    start_time = time.time()
    print ('Writing data to disk ...', end='', flush=True)

    np.savetxt('output_intestine/sdge_' + str(num_cells) + '_cells_'
               + str(locations.shape[0]) + '_locations.txt', sdge, fmt='%.4e')

    print ('done (', round(time.time()-start_time, 2), 'seconds )')

    ###########################
    # 6. Plot gene expression #
    ###########################

    gene_list_to_plot = ['Glul', 'Cyp2el', 'Cyp2f2', 'Alb']
    plot_gene_patterns_1D(locations, sdge, gene_list_to_plot, 
                          folder='output_intestine/', gene_names=gene_names, 
                          num_cells)

    ###################################
    # 7. Correlate results with FISH data #
    ###################################
    
    with open('output_intestine/results.txt', 'a') as f:
        f.write('number_cells,' +  ','.join(gene_names) + '\n')
        f.write(str(num_cells) + ',')
        for i in range(len(gene_names)):
            f.write(str(round(pearsonr(sdge[i, :], dge[:, i])[0], 2)) + ',')


