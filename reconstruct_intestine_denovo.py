#########
# about #
#########

__version__ = "0.1.1"
__author__ = ["Nikos Karaiskos", "Mor Nitzan"]
__status__ = "beta"
__licence__ = "GPL"
__email__ = ["nikolaos.karaiskos@mdc-berlin.de", "mornitzan@fas.harvard.edu"]

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

    # Read the intestine data
    zfile = zipfile.ZipFile('datasets/intestine/dge.tsv.zip')
    zfile.extract('dge.tsv')
    gene_names = np.genfromtxt('dge.tsv', usecols=0, dtype='str', skip_header=1)

    locations_original = np.loadtxt('datasets/intestine/zones.tsv',skiprows=1,usecols=range(1,4))
    locations_original = locations_original[:,2]
    grid_len = len(np.unique(locations_original))
    
    dge = np.loadtxt('dge.tsv',skiprows=1,usecols=range(1,1384))
    dge_full = dge.T
     
    # Optional: downsample number of cells
    num_cells = dge_full.shape[0] # all cells are used
    cells_selected = np.random.choice(dge_full.shape[0], num_cells, replace=False)
    dge_full = dge_full[cells_selected, :]    
    locations_original = locations_original[cells_selected]

    # Compute mean dge over original zones 
    dge_full_mean = np.zeros((grid_len,dge_full.shape[1]))
    for i in range(grid_len):
        indices =  np.argwhere(locations_original==i).flatten()
        temp = np.mean(dge_full[indices,:],axis=0)
        dge_full_mean[i,:] = temp
    dge_full_mean = dge_full_mean.T 
    
    # Select variable genes
    var_genes = np.argsort(np.divide(np.var(dge_full.T,axis=1),np.mean(dge_full.T,axis=1)+0.0001))
    dge = dge_full[:,var_genes[-100:]]  
        
    print ('done (', round(time.time()-start_time, 2), 'seconds )')
    
    ################################
    # 2. Set the target space grid #
    ################################

    print ('Reading the target space ... ', end='', flush=True)    
    # Read and use a 1d grid
    
    locations = np.vstack((range(grid_len),np.ones(grid_len))).T
    
    print ('done')

    ######################################
    # 3. Setup for the OT reconstruction #
    ######################################
    
    cost_expression, cost_locations = setup_for_OT_reconstruction(dge, locations, 
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

    # Compute sdge
    sdge = np.dot(dge_full.T, gw)

    # Compute mean expression distribution over embedded zones 
    mean_exp_new_dist = np.zeros((grid_len,grid_len))
    for i in range(grid_len):
        indices =  np.argwhere(locations_original==i).flatten()
        temp = np.sum(gw[indices,:],axis=0)
        mean_exp_new_dist[i,:] = temp/np.sum(temp)

    print (' ... done (', round(time.time()-start_time, 2), 'seconds )')

    #########################################
    # 5. Write data to disk for further use #
    #########################################

    start_time = time.time()
    print ('Writing data to disk ...', end='', flush=True)

    np.savetxt('output_intestine/sdge_' + str(num_cells) + '_cells_'
               + str(locations.shape[0]) + '_locations.txt', sdge, fmt='%.4e')

    print ('done (', round(time.time()-start_time, 2), 'seconds )')

    ###########################################################################################
    # 6. Plot histogram showing the distribution over embedded zones for each original zone #
    ###########################################################################################

    plot_histogram_intestine(mean_exp_new_dist, folder='output_intestine/')
    
    ###########################################################################################
    # 7. Plot spatial expression of a few gene groups for the original and embedded zones #
    ###########################################################################################

    plot_spatial_expression_intestine(dge_full_mean, sdge, gene_names, folder='output_intestine/')
    
    
