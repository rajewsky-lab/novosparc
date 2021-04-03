import numpy as np
import os
import novosparc
from scipy.spatial.distance import cdist
from contextlib import redirect_stdout
import io
import pandas as pd
import operator

class Tissue():
	"""The class that handles the processes for the tissue reconstruction. It is responsible for keeping
	the data, creating the reconstruction and saving the results."""

	def __init__(self, dataset, locations, atlas_matrix=None, markers_to_use=None, output_folder=None):
		"""Initialize the tissue using the dataset and locations.
		dataset -- Anndata object for the single cell data (cells x genes)
		locations -- target space locations (locations x dimensions)
		atlas_matrix -- optional atlas matrix (atlas locations x markers)
		markers_to_use -- optional indices of atlas marker genes in dataset
		output_folder -- folder path to save the plots and data"""
		self.dataset = dataset
		self.dge = dataset.X
		self.locations = locations
		self.num_cells = len(dataset.obs)
		self.num_locations = locations.shape[0]
		self.gene_names = np.array(dataset.var.index.tolist())
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


	def setup_smooth_costs(self, dge_rep=None, num_neighbors_s=5, num_neighbors_t=5,
						   locations_metric='minkowski', locations_metric_p=2,
						   expression_metric='minkowski', expression_metric_p=2, verbose=True):
		"""
		Set cell-cell expression cost and location-location physical distance cost
		dge_rep -- some representation of the expression matrix, e.g. pca, selected highly variable genes etc.
		num_neighbors_s -- num neighbors for cell-cell expression cost
		num_neighbors_t -- num neighbors for location-location physical distance cost
		locations_metric -- discrepancy metric - physical distance cost
		locations_metric_p -- power parameter of the Minkowski metric - locations distance cost
		expression_metric -- discrepancy metric - expression distance cost
		expression_metric_p -- power parameter of the Minkowski metric - expression distance cost
		"""
		dge_rep = dge_rep if dge_rep is not None else self.dge
		self.costs['expression'], self.costs['locations'] = novosparc.rc.setup_for_OT_reconstruction(dge_rep,
																									 self.locations,
																									 num_neighbors_source=num_neighbors_s,
																									 num_neighbors_target=num_neighbors_t,
																									 locations_metric=locations_metric, locations_metric_p=locations_metric_p,
																									 expression_metric=expression_metric, expression_metric_p=expression_metric_p,
																									 verbose=verbose)

	def setup_linear_cost(self, markers_to_use=None, atlas_matrix=None, markers_metric='euclidean', markers_metric_p=2):
		"""
		Set linear(=atlas) cost matrix
		markers_to_use -- indices of the marker genes
		atlas_matrix -- corresponding reference atlas
		markers_metric -- discrepancy metric - cell-location distance cost
		markers_metric_p -- power parameter of the Minkowski metric - cell-location distance cost
		"""
		self.atlas_matrix = atlas_matrix if atlas_matrix is not None else self.atlas_matrix
		self.markers_to_use = markers_to_use if markers_to_use is not None else self.markers_to_use

		cell_expression = self.dge[:, self.markers_to_use] / np.amax(self.dge[:, self.markers_to_use])
		atlas_expression = self.atlas_matrix / np.amax(self.atlas_matrix)

		self.costs['markers'] = cdist(cell_expression, atlas_expression, metric=markers_metric, p=markers_metric_p)
		self.num_markers = len(self.markers_to_use)

	def setup_reconstruction(self, markers_to_use=None, atlas_matrix=None, num_neighbors_s=5, num_neighbors_t=5, verbose=True):
		"""
		Set cost matrices for reconstruction. If there are marker genes and an reference atlas matrix, these
		can be used as well.
		markers_to_use -- indices of the marker genes
		atlas_matrix -- reference atlas corresponding to markers_to_use
		num_neighbors_s -- num neighbors for cell-cell expression cost
		num_neighbors_t -- num neighbors for location-location physical distance cost
		"""
		self.atlas_matrix = atlas_matrix if atlas_matrix is not None else self.atlas_matrix
		self.markers_to_use = markers_to_use if markers_to_use is not None else self.markers_to_use

		# calculate cost matrices for OT
		if self.markers_to_use is not None:
			self.setup_linear_cost(self.markers_to_use, self.atlas_matrix)
		self.setup_smooth_costs(num_neighbors_s=num_neighbors_s, num_neighbors_t=num_neighbors_t,verbose=verbose)

	def reconstruct(self, alpha_linear, epsilon=5e-4, p_locations=None, p_expression=None,
					search_epsilon=True, random_ini=False, verbose=True):
		"""Reconstruct the tissue using the calculated costs and the given alpha value
		alpha_linear -- this is the value the set the weight of the reference atlas if there is any
		epsilon -- coefficient of entropy regularization
		p_locations -- marginal probability of locations
		p_expression -- marginal probability of cells
		search_epsilon -- run with increased epsilon if numerical errors occur
		random_ini -- random initialization of transportation matrix for stochastic results
		"""
		if verbose:
			print ('Reconstructing spatial information with', self.num_markers,
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

		while first_pass or (epsilon < max_epsilon):
			f = io.StringIO()

			print('Trying with epsilon: ' + '{:.2e}'.format(epsilon))
			with redirect_stdout(f):
				gw = novosparc.rc._GWadjusted.gromov_wasserstein_adjusted_norm(cost_marker_genes, cost_expression, cost_locations,
												  self.alpha_linear, self.p_expression, self.p_locations,
												  'square_loss', epsilon=epsilon, verbose=verbose, random_ini=random_ini)
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
		sdge = np.dot(self.dge.T, gw)
		self.gw = gw
		self.sdge = sdge

	def calculate_sdge_for_all_genes(self):
		raw_data = self.dataset.raw.to_adata()
		dge_full = raw_data.X
		sdge_full = np.dot(dge_full.T, self.gw)
		return sdge_full

	def calculate_spatially_informative_genes(self, selected_genes=None):
		"""Calculate spatially informative genes using Moran's I
		selected_genes -- subset of genes to check. if None, calculate for every gene
		"""
		if selected_genes is not None:
			selected_genes = np.asarray(selected_genes)
			gene_indices = np.nonzero(np.in1d(self.gene_names, selected_genes))[0]
			sdge = self.sdge[gene_indices, :]
			gene_names = selected_genes
		else:
                        gene_names = self.gene_names
                        sdge = self.sdge

		num_genes = sdge.shape[0]
		print('Morans I analysis for %i genes...' % num_genes, end='', flush=True)
		dataset = pd.DataFrame(sdge.T)
		mI, pvals = novosparc.analysis._analysis.get_moran_pvals(dataset, self.locations, n_neighbors=8)
		mI = np.array(mI)
		mI[np.isnan(mI)] = -np.inf
		important_gene_ids = np.argsort(mI)[::-1]
		important_gene_names = gene_names[important_gene_ids]
		results = pd.DataFrame({'genes': gene_names, 'mI':mI, 'pval':pvals})
		results = results.sort_values(by=['mI'], ascending=False)
	
		self.spatially_informative_genes = results
