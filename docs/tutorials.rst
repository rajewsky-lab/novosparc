General usage 
=============
To spatially reconstruct gene expression, ``novoSpaRc`` performs the following
steps:

1. Read the gene expression matrix.
   
   a. *Optional*: select a random set of cells for the reconstruction.
   b. *Optional*: select a small set of genes (e.g. highly variable).
2. Construct the target space.
3. Setup the optimal transport reconstruction.

   a. *Optional*: use existing information of marker genes, if available.
4. Perform the spatial reconstruction.

   a. Assign cells a probability distribution over the target space.
   b. Derive a virtual in situ hybridization (vISH) for all genes over the target space.

5. Write outputs to file for further use, such as the spatial gene expression matrix and the target space coordinates.
6. *Optional*: plot spatial gene expression patterns.
7. *Optional*: identify and plot spatial archetypes.

Demonstration
~~~~~~~~~~~~~
We provide scripts that spatially reconstruct two of the tissues presented
in the paper: the intestinal epithelium [Moor18]_ and the stage 6 Drosophila embryo
[BDTNP]_. 

See also our `tutorial <https://github.com/rajewsky-lab/novosparc/blob/master/reconstruct_drosophila_embryo_tutorial.ipynb>`_ on reconstructing the Drosophila embryo.

The intestinal epithelium
~~~~~~~~~~~~~~~~~~~~~~~~~
The ``reconstruct_intestine_denovo.py`` script reconstructs the crypt-to-villus axis of the mammalian intestinal epithelium, based on data from [Moor18]_. 
The reconstruction is performed *de novo*, without using any marker genes. 
The script outputs plots of (a) a histogram showing the distribution of assignment values over embedded zones for each original villus zone, and (b) average spatial gene expression over the original villus zones and embedded zones of 4 gene groups.

Running time on a standard computer is under a minute.

The *Drosophila* embryo
~~~~~~~~~~~~~~~~~~~~~~~
The ``reconstruct_bdtnp_with_markers.py`` script reconstructs the early
*Drosophila* embryo with only a handful of markers, based on the [BDTNP]_ dataset. 
All cells are used and
a random set of 1-4 markers is selected. The script outputs plots of
gene expression for a list of genes, as well as Pearson correlations of the
reconstructed and original expression values for all genes.
Notice that the results depend on which marker genes are selected. 
In the manuscript we averaged the results over many different choices of marker genes.

Running time on a standard desktop computer is around 6-7 minutes.

Running novoSpaRc on your data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A template file for running ``novoSpaRc`` on custom datasets is 
provided (``reconstruct_tissue.py``). To successfully run ``novoSpaRc`` modify the
template file accordingly.

Constructing different grid shapes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We advise to use ```novoSpaRc`` with diverse target spaces to assess how robust
the spatial reconstructions are. A straightforward way to create a target space
which is more interesting than a square grid, is to have a simple image with the
target space painted in black on it, such as the one below:

.. image:: https://raw.githubusercontent.com/nukappa/nukappa.github.io/master/images/tissue_example.png
   :width: 200px
   :align: center

Then use the function ``create_target_space_from_image`` from the geometry module
to read the image and create a target space out of it. It is advisable to
sample a number of all the read locations and not use them all.

