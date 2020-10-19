import numpy as np
import os
import novosparc
from scipy.spatial.distance import cdist


from io import StringIO
import sys

class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


class Tissue():
	"""The class that handles the processes for the tissue reconstruction. It is responsible for keeping
	the data, creating the reconstruction and saving the results."""

	def __init__(self, dataset, locations, output_folder=None):
		"""Initialize the tissue using the dataset and locations.
		dataset -- Anndata object for the single cell data
		locations -- target space locations
		atlas_matrix -- optional atlas matrix
		output_folder -- folder path to save the plots and data"""
		self.dataset = dataset
		self.dge = dataset.X
		self.locations = locations
		self.num_cells = len(dataset.obs)
		self.num_locations = locations.shape[0]
		self.gene_names = np.array(dataset.var.index.tolist())
		self.atlas_matrix = atlas_matrix

		# if the output folder does not exist, create one
		if output_folder is not None and not os.path.exists(output_folder):
			os.mkdir(output_folder)
		self.output_folder = output_folder

		self.num_markers = 0
		self.costs = {'expression': np.ones((self.num_cells, self.num_cells)),
					  'locations': np.ones((self.num_locations, self.num_locations)),
					  'markers': np.ones((self.num_cells, self.num_locations))}
		self.gw = None
		self.sdge = None
		self.spatially_informative_genes = None

	def setup_smooth_costs(self, dge_rep=None, num_neighbors_s=5, num_neighbors_t=5, verbose=True):
		"""
		Set cell-cell expression cost and location-location physical distance cost
		dge_rep -- some representation of the expression matrix, e.g. pca, selected highly variable genes etc.
		num_neighbors_s -- num neighbors for cell-cell expression cost
		num_neighbors_t -- num neighbors for location-location physical distance cost
		"""
		dge_rep = dge_rep if dge_rep is not None else self.dge
		self.costs['expression'], self.costs['locations'] = novosparc.rc.setup_for_OT_reconstruction(dge_rep,
																			   self.locations,
																			   num_neighbors_source = num_neighbors_s,
																			   num_neighbors_target = num_neighbors_t,
																			  verbose=verbose)

	def setup_linear_cost(self, markers_to_use, insitu_matrix):
		"""
		Set linear(=atlas) cost matrix
		markers_to_use -- indices of the marker genes
		insitu_matrix -- corresponding reference atlas
		"""
		self.costs['markers'] = cdist(self.dge[:, markers_to_use]/np.amax(self.dge[:, markers_to_use]),
						  insitu_matrix/np.amax(insitu_matrix))
		self.num_markers = len(markers_to_use)



	def setup_reconstruction(self, markers_to_use=None, insitu_matrix=None, num_neighbors_s=5, num_neighbors_t=5, verbose=True):
		"""
		Set cost matrices for reconstruction. If there are marker genes and an reference atlas matrix, these
		can be used as well.
		markers_to_use -- indices of the marker genes
		insitu_matrix -- reference atlas corresponding to markers_to_use
		num_neighbors_s -- num neighbors for cell-cell expression cost
		num_neighbors_t -- num neighbors for location-location physical distance cost
		"""
		if markers_to_use is not None:
			self.setup_linear_cost(markers_to_use, insitu_matrix)

		# calculate cost matrices for OT
		if self.costs['expression'] is None or self.costs['locations'] is None:
			self.setup_smooth_costs(num_neighbors_s=num_neighbors_s,
									num_neighbors_t=num_neighbors_t,
									verbose=verbose)


	def reconstruct(self, alpha_linear, epsilon=5e-4, verbose=True, **kwargs):
		"""Reconstruct the tissue using the calculated costs and the given alpha value
		alpha_linear -- this is the value the set the weight of the reference atlas if there is any
		"""
		if verbose:
			print ('Reconstructing spatial information with', self.num_markers,
			   'markers:', self.num_cells, 'cells and',
			   self.num_locations, 'locations ... ')

		# Distributions at target and source spaces
		p_locations, p_expression = novosparc.rc.create_space_distributions(self.num_locations, self.num_cells)

		cost_marker_genes = self.costs['markers']
		cost_expression = self.costs['expression']
		cost_locations = self.costs['locations']

		# get lowest epsilon
		ini_epsilon = epsilon
		max_epsilon = 5e-1
		mult_fac = 10
		stopped_iter_zero = True
		warning_msg = 'Warning: numerical errors at iteration '
		while (epsilon < max_epsilon) and (stopped_iter_zero):
			print('Trying with epsilon: ' + '{:.2e}'.format(epsilon))
			with Capturing() as output:
				gw = novosparc.rc._GWadjusted.gromov_wasserstein_adjusted_norm(cost_marker_genes, cost_expression,
																			   cost_locations, alpha_linear,
																			   p_expression, p_locations,
																			   'square_loss', epsilon=epsilon,
																			   verbose=verbose, **kwargs)
			iter_stop = [int(s.split(warning_msg)[1]) for s in np.unique(output) if warning_msg in s]
			stopped_iter_zero = (len(output) > 0) and (np.all(np.array(iter_stop) == 0))
			epsilon = epsilon * mult_fac

		print('\n'.join(output))
		if epsilon > ini_epsilon:
			print('Using epsilon: %.08f' % (epsilon / mult_fac))

		sdge = np.dot(self.dge.T, gw)
		self.gw = gw
		self.sdge = sdge

	def calculate_sdge_for_all_genes(self):
		raw_data = self.dataset.raw.to_adata()
		dge_full = raw_data.X
		sdge_full = np.dot(dge_full.T, self.gw)
		return sdge_full

	def calculate_spatially_informative_genes(self, selected_genes=None):
		"""
		Calculate spatially informative genes using Moran's I
		selected_genes -- subset of genes to check. if None, calculate for every gene
		"""
		if selected_genes == None:
			selected_genes = self.gene_names
		important_gene_names = novosparc.analysis.morans(self.sdge, self.gene_names, self.locations, folder=self.output_folder, selected_genes=selected_genes)
		self.spatially_informative_genes = important_gene_names
