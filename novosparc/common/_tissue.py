import numpy as np
import os
import novosparc
from scipy.spatial.distance import cdist


class Tissue():
	"""The class that handles the processes for the tissue reconstruction. It is responsible for keeping
	the data, creating the reconstruction and saving the results."""

	def __init__(self, dataset, locations, output_folder=None):
		"""Initialize the tissue using the dataset and locations.
		dataset -- Anndata object for the single cell data
		locations -- target space locations
		output_folder -- folder path to save the plots and data"""
		self.dataset = dataset
		self.dge = dataset.X
		self.locations = locations
		self.num_cells = len(dataset.obs)
		self.num_locations = locations.shape[0]
		self.gene_names = np.array(dataset.var.index.tolist())

		# if the output folder does not exist, create one
		if output_folder is not None and not os.path.exists(output_folder):
			os.mkdir(output_folder)
		self.output_folder = output_folder

		self.num_markers = 0
		self.costs = None
		self.gw = None
		self.sdge = None
		self.spatially_informative_genes = None

	def setup_reconstruction(self, markers_to_use=None, insitu_matrix=None, num_neighbors_s=5, num_neighbors_t=5):
		"""Setup cost matrices for reconstruction. If there are marker genes and an reference atlas matrix, these
		can be used as well.
		markers_to_use -- indices of the marker genes
		insitu_matrix -- reference atlas
		num_neighbors_s -- number of neighbors of the source for OT setup
		num_neighbors_t -- number of neighbors of the target for OT setup
		"""

		# if there are no markers, keep the dge as it is and set ones as the marker costs
		if markers_to_use is None:
			cost_marker_genes = np.ones((self.num_cells, self.num_locations))
			dge = self.dge
		# if there are marker genes, calculate the cost
		else:
			cost_marker_genes = cdist(self.dge[:, markers_to_use]/np.amax(self.dge[:, markers_to_use]),
							  insitu_matrix/np.amax(insitu_matrix))
			dge = self.dge[:, np.setdiff1d(np.arange(self.dge.shape[1]), markers_to_use)]
			self.num_markers = len(markers_to_use)

		# calculate cost matrices for OT
		cost_expression, cost_locations = novosparc.rc.setup_for_OT_reconstruction(dge,
																			   self.locations,
																			   num_neighbors_source = num_neighbors_s,
																			   num_neighbors_target = num_neighbors_t)

		costs = {'expression':cost_expression,'locations': cost_locations,'markers': cost_marker_genes}
		self.costs = costs

	def reconstruct(self, alpha_linear, epsilon=5e-4):
		"""Reconstruct the tissue using the calculated costs and the given alpha value
		alpha_linear -- this is the value the set the weight of the reference atlas if there is any
		"""
		print ('Reconstructing spatial information with', self.num_markers,
           'markers:', self.num_cells, 'cells and',
           self.num_locations, 'locations ... ')

		# Distributions at target and source spaces
		p_locations, p_expression = novosparc.rc.create_space_distributions(self.num_locations, self.num_cells)

		cost_marker_genes = self.costs['markers']
		cost_expression = self.costs['expression']
		cost_locations = self.costs['locations']

		gw = novosparc.rc._GWadjusted.gromov_wasserstein_adjusted_norm(cost_marker_genes, cost_expression, cost_locations,
												  alpha_linear, p_expression, p_locations,
												  'square_loss', epsilon=epsilon, verbose=True)
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
		if selected_genes == None:
			selected_genes = self.gene_names
		important_gene_names = novosparc.analysis.morans(self.sdge, self.gene_names, self.locations, folder=self.output_folder, selected_genes=selected_genes)
		self.spatially_informative_genes = important_gene_names



