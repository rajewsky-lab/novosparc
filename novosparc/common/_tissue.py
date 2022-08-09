import numpy as np
import os
import novosparc
from scipy.spatial.distance import cdist
from scipy.stats import zscore
import scipy
from sklearn.mixture import BayesianGaussianMixture
from contextlib import redirect_stdout
import io
import pandas as pd
import operator


class Tissue():
    """The class that handles the processes for the tissue reconstruction. It is responsible for keeping
	the data, creating the reconstruction and saving the results."""

    def __init__(self, dataset, locations, atlas_matrix=None, markers_to_use=None, output_folder=None):
        """Initialize the tissue using the dataset and locations.
		dataset        -- Anndata object for the single cell data (cells x genes)
		locations      -- target space locations (locations x dimensions)
		atlas_matrix   -- optional atlas matrix (atlas locations x markers)
		markers_to_use -- optional indices of atlas marker genes in dataset
		output_folder  -- folder path to save the plots and data"""
        self.dataset = dataset
        self.dge = dataset.X
        self.locations = locations
        self.num_cells = len(dataset.obs)
        self.num_locations = locations.shape[0]
        self.gene_names = np.array(dataset.var.index.tolist())
        self.num_genes = len(self.gene_names)
        self.atlas_matrix = atlas_matrix
        self.markers_to_use = markers_to_use

        # if the output folder does not exist, create one
        if output_folder is not None and not os.path.exists(output_folder):
            os.mkdir(output_folder)
        self.output_folder = output_folder

        self.num_markers = 0 if markers_to_use is None else len(markers_to_use)
        self.costs = None
        self.gw = None
        self.sdge = None
        self.spatially_informative_genes = None
        self.p_expression = None
        self.p_locations = None
        self.costs = {'expression': np.ones((self.num_cells, self.num_cells)),
                      'locations': np.ones((self.num_locations, self.num_locations)),
                      'markers': np.ones((self.num_cells, self.num_locations))}
        self.cleaned_dge = None

    def setup_smooth_costs(self, dge_rep=None, num_neighbors_s=5, num_neighbors_t=5,
                           locations_metric='minkowski', locations_metric_p=2,
                           expression_metric='minkowski', expression_metric_p=2, verbose=True):
        """
		Set cell-cell expression cost and location-location physical distance cost
		dge_rep             -- some representation of the expression matrix, e.g. pca, selected highly variable genes etc.
		num_neighbors_s     -- num neighbors for cell-cell expression cost
		num_neighbors_t     -- num neighbors for location-location physical distance cost
		locations_metric    -- discrepancy metric - physical distance cost
		locations_metric_p  -- power parameter of the Minkowski metric - locations distance cost
		expression_metric   -- discrepancy metric - expression distance cost
		expression_metric_p -- power parameter of the Minkowski metric - expression distance cost
		"""
        dge_rep = dge_rep if dge_rep is not None else self.dge
        self.costs['expression'], self.costs['locations'] = novosparc.rc.setup_for_OT_reconstruction(dge_rep,
                                                                                                     self.locations,
                                                                                                     num_neighbors_source=num_neighbors_s,
                                                                                                     num_neighbors_target=num_neighbors_t,
                                                                                                     locations_metric=locations_metric,
                                                                                                     locations_metric_p=locations_metric_p,
                                                                                                     expression_metric=expression_metric,
                                                                                                     expression_metric_p=expression_metric_p,
                                                                                                     verbose=verbose)

    def setup_linear_cost(self, markers_to_use=None, atlas_matrix=None, markers_metric='minkowski', markers_metric_p=2):
        """
		Set linear(=atlas) cost matrix
		markers_to_use   -- indices of the marker genes
		atlas_matrix     -- corresponding reference atlas
		markers_metric   -- discrepancy metric - cell-location distance cost
		markers_metric_p -- power parameter of the Minkowski metric - cell-location distance cost
		"""
        self.atlas_matrix = atlas_matrix if atlas_matrix is not None else self.atlas_matrix
        self.markers_to_use = markers_to_use if markers_to_use is not None else self.markers_to_use

        cell_expression = self.dge[:, self.markers_to_use] / np.amax(self.dge[:, self.markers_to_use])
        atlas_expression = self.atlas_matrix / np.amax(self.atlas_matrix)

        self.costs['markers'] = cdist(cell_expression, atlas_expression, metric=markers_metric, p=markers_metric_p)
        self.num_markers = len(self.markers_to_use)

    def setup_reconstruction(self, markers_to_use=None, atlas_matrix=None, num_neighbors_s=5, num_neighbors_t=5,
                             verbose=True):
        """
		Set cost matrices for reconstruction. If there are marker genes and a reference atlas matrix, these
		can be used as well.
		markers_to_use  -- indices of the marker genes
		atlas_matrix    -- reference atlas corresponding to markers_to_use
		num_neighbors_s -- num neighbors for cell-cell expression cost
		num_neighbors_t -- num neighbors for location-location physical distance cost
		"""
        self.atlas_matrix = atlas_matrix if atlas_matrix is not None else self.atlas_matrix
        self.markers_to_use = markers_to_use if markers_to_use is not None else self.markers_to_use

        # calculate cost matrices for OT
        if self.markers_to_use is not None:
            self.setup_linear_cost(self.markers_to_use, self.atlas_matrix)
        self.setup_smooth_costs(num_neighbors_s=num_neighbors_s, num_neighbors_t=num_neighbors_t, verbose=verbose)

    def reconstruct(self, alpha_linear, epsilon=5e-4, p_locations=None, p_expression=None,
                    search_epsilon=True, random_ini=False, verbose=True):
        """Reconstruct the tissue using the calculated costs and the given alpha value
		alpha_linear   -- this is the value the set the weight of the reference atlas if there is any
		epsilon        -- coefficient of entropy regularization
		p_locations    -- marginal probability of locations
		p_expression   -- marginal probability of cells
		search_epsilon -- run with increased epsilon if numerical errors occur
		random_ini     -- random initialization of transportation matrix for stochastic results
		"""
        if verbose:
            print('Reconstructing spatial information with', self.num_markers,
                  'markers:', self.num_cells, 'cells and',
                  self.num_locations, 'locations ... ')

        # Distributions at target and source spaces
        p_locations_c, p_expression_c = novosparc.rc.create_space_distributions(self.num_locations, self.num_cells)
        self.p_locations = p_locations_c if p_locations is None else p_locations
        self.p_expression = p_expression_c if p_expression is None else p_expression

        self.alpha_linear = alpha_linear

        cost_marker_genes = self.costs['markers']
        cost_expression = self.costs['expression']
        cost_locations = self.costs['locations']

        ini_epsilon = epsilon
        max_epsilon = 5e-1
        mult_fac = 10
        warning_msg = 'Warning: numerical errors at iteration '
        first_pass = True

        # docu ToDo: What's happening here?
        while first_pass or (epsilon < max_epsilon):
            f = io.StringIO()

            print('Trying with epsilon: ' + '{:.2e}'.format(epsilon))
            with redirect_stdout(f):
                gw = novosparc.rc._GWadjusted.gromov_wasserstein_adjusted_norm(cost_marker_genes, cost_expression,
                                                                               cost_locations,
                                                                               self.alpha_linear, self.p_expression,
                                                                               self.p_locations,
                                                                               'square_loss', epsilon=epsilon,
                                                                               verbose=verbose, random_ini=random_ini)
            out = f.getvalue()

            # docu ToDo: What's happening here?
            if warning_msg not in out:
                f.close()
                break
            else:
                epsilon = epsilon * mult_fac
                f.close()

            if not search_epsilon:
                break

            first_pass = False

        while first_pass or (epsilon < max_epsilon):
            f = io.StringIO()

            print('Trying with epsilon: ' + '{:.2e}'.format(epsilon))
            with redirect_stdout(f):
                gw = novosparc.rc._GWadjusted.gromov_wasserstein_adjusted_norm(cost_marker_genes, cost_expression,
                                                                               cost_locations,
                                                                               self.alpha_linear, self.p_expression,
                                                                               self.p_locations,
                                                                               'square_loss', epsilon=epsilon,
                                                                               verbose=verbose, random_ini=random_ini)
            out = f.getvalue()

            if warning_msg not in out:
                f.close()
                break
            else:
                epsilon = epsilon * mult_fac
                f.close()

            if not search_epsilon:
                break

            first_pass = False

        if epsilon > ini_epsilon:
            epsilon = (epsilon / mult_fac)
            print('Using epsilon: %.08f' % epsilon)

        self.epsilon = epsilon

        if type(self.dge) is scipy.sparse.csr_matrix:
            self.dge = scipy.sparse.csr_matrix.toarray(self.dge)

        sdge = np.dot(self.dge.T, gw)
        self.gw = gw
        self.sdge = sdge

    # When is this used? Or what is it for?
    def calculate_sdge_for_all_genes(self):
        raw_data = self.dataset.raw.to_adata()
        dge_full = raw_data.X
        sdge_full = np.dot(dge_full.T, self.gw)
        return sdge_full

    def calculate_spatially_informative_genes(self, selected_genes=None, n_neighbors=8):
        """Calculate spatially informative genes using Moran's I
		selected_genes -- subset of genes to check. if None, calculate for every gene
		"""
        if selected_genes is not None:
            selected_genes = np.asarray(selected_genes)
            selected_genes = np.unique(selected_genes)
            # gene_indices = np.nonzero(np.in1d(self.gene_names, selected_genes))[0]
            gene_indices = pd.DataFrame(np.arange(self.num_genes), index=self.gene_names)[0].loc[selected_genes].values
            sdge = self.sdge[gene_indices, :]
            gene_names = selected_genes
        else:
            gene_names = self.gene_names
            sdge = self.sdge

        num_genes = sdge.shape[0]
        print('Morans I analysis for %i genes...' % num_genes, end='', flush=True)
        mI, pvals = novosparc.an.get_moran_pvals(sdge.T, self.locations, n_neighbors=n_neighbors)
        mI = np.array(mI)
        mI[np.isnan(mI)] = -np.inf
        results = pd.DataFrame({'genes': gene_names, 'mI': mI, 'pval': pvals})
        results = results.sort_values(by=['mI'], ascending=False)

        self.spatially_informative_genes = results

    def cleaning_expression_data(self, dataset=None, expression_matrix=None, normalization=None, cov_prior=None,
                                 selected_genes=None, plotting=None):

        # TODO: not sure if it's proper/makes sense to use the anndata dataset here already. Probably not. But it's
        #  the version I have right now for creating a gene subset. Has to be change later if necessary

        """
        :param dataset              -- Scanpy AnnData with 'spatial' matrix in obsm containing the spatial coordinates of the tissue
        :param expression_matrix:   -- either dge or sdge
        :param normalization:       -- reconstructed data has to be normalized first, raw-data too if not previously
                                       done, choose from 'minmax', 'log', 'zscore'
        :param cov_prior:           -- change to widen the applied fit curves, e.g. to capture relevant
                                       low-expression genes, has to be given in the form: [(#,)]
        :param selected_genes:      -- subset of genes to check. if None, calculate for every gene
        :param plotting:            -- when list of genes given plot cntrl plot with mapping before and after filtering
        :return:                    -- tissue object with cleaned expression matrix
        """

        # check normalization method provided
        possible_normalization = ['minmax', 'log', 'zscore', None]
        if normalization not in possible_normalization:
            raise ValueError("Invalid normalization method. Expected one of: %s" % possible_normalization)

        # subset matrix
        if selected_genes is None:
            used_matrix = expression_matrix
        elif isinstance(selected_genes, list) & len(selected_genes) >= 1:
            subset_cols = []
            for i, gene in enumerate(selected_genes):
                if gene in dataset.var_names:
                    subset_cols.append(np.asarray(dataset[:, gene].X).reshape(-1, 1))
            used_matrix = np.concatenate(subset_cols, axis=1)
        else:
            raise ValueError("Invalid input for selected_genes. When given then it has to be a list with genes that"
                             "should be tested. Else give non and cleaning will performed on the whole matrix.")

        # normalize data
        if normalization == 'minmax':
            uncleaned_matrix = (used_matrix - np.min(used_matrix)) / \
                               (np.max(used_matrix) - np.min(used_matrix))
        elif normalization == 'log':
            uncleaned_matrix = np.log(used_matrix)
        elif normalization == 'zscore':
            uncleaned_matrix = zscore(used_matrix)
        else:
            uncleaned_matrix = used_matrix

        # transform to a pd dict for faster iteration
        # TODO there should be a check for applying the transposion or not depending on wether the input is transposed or not
        uncleaned_matrix_dict = pd.DataFrame(uncleaned_matrix.T).to_dict('records')

        # apply model and filtering
        modded_cols = []

        for row in uncleaned_matrix_dict:

            # transform to array
            expression_values = np.asarray(list(row.values())).reshape(-1, 1)

            # apply model
            if cov_prior is None:
                gmm = BayesianGaussianMixture(n_components=2).fit(expression_values)
            else:
                # widen the fitted curves to include more expression values of the defaults distributions edges
                gmm = BayesianGaussianMixture(n_components=2,
                                              covariance_prior=cov_prior,
                                              ).fit(expression_values)
            # get labels for distributions
            labels = gmm.predict(expression_values)

            # merge labels column with original expression value column
            label_assignment = np.concatenate((expression_values,
                                               labels.reshape(-1, 1)), axis=1)

            # check how many labels and how many values per label
            vl_cnts = pd.Series(labels).value_counts()

            # TODO: I would like to write this stuff without those magic numbers (if even possible in python?)
            # only apply sorting when 2 distributions where modelled
            if len(vl_cnts) > 1:
                # when the 0 dist is the dist of choice, labels have to be inverted for multiplication
                if vl_cnts[0] < vl_cnts[1]:
                    # invert labels
                    label_assignment[:, 1] = np.logical_not(label_assignment[:, 1]).astype(int)
                # multiply expression values with label values so that the 0 dist values are effectively removed
                label_assignment[:, 0] *= label_assignment[:, 1]

            # build a list of modified columns
            modded_cols.append(label_assignment[:, 0].reshape(-1, 1))

        modded_matrix = np.concatenate(modded_cols, axis=1)

        # in case of subset was used, update the subsetted columns in the original data and return the full expression matrix
        if selected_genes is not None:
            try:
                df_expression_matrix = pd.DataFrame(expression_matrix.T, columns=dataset.var_names)
            except ValueError:
                df_expression_matrix = pd.DataFrame(expression_matrix, columns=dataset.var_names)
            df_modded_matrix = pd.DataFrame(modded_matrix, columns=selected_genes)
            df_expression_matrix.update(df_modded_matrix)

            modded_matrix_full = df_expression_matrix.to_numpy()
        else:
            modded_matrix_full = modded_matrix

        # optional plotting of cleaning results
        # TODO have to think a bit more about this one how to do it properly (e.g. when/how to use the novosparc also
        #  not sure if this is really necessary when we have the option of using a subset - maybe just plotting manually is cleaner?

        self.cleaned_dge = modded_matrix_full
