# novoSpaRc - de novo Spatial Reconstruction of single-cell gene expression

## About
This package is created and maintained by 
[Nikos Karaiskos](mailto:nikolaos.karaiskos@mdc-berlin.de) and
[Mor Nitzan](mailto:mornitzan@fas.harvard.edu). 
`novoSpaRc` can be used to predict the locations of cells
in space by using single-cell RNA sequencing data. An existing reference
database of marker genes is not required, but enhances mappability if
available.

`novoSpaRc` accompanies the following preprint

*Charting tissues from single-cell transcriptomes*, <br />
[*bioRxiv (2018)*](https://www.biorxiv.org/content/early/2018/10/30/456350)

M. Nitzan<sup>#</sup>, N. Karaiskos<sup>#</sup>,
N. Friedman<sup>&</sup> and N. Rajewsky<sup>&</sup>

<sup>#</sup> Contributed equally <br />
<sup>&</sup> Corresponding authors: 
[N. Friedman](mailto:nir.friedman@mail.huji.ac.il), 
[N.Rajewsky](mailto:rajewsky@mdc-berlin.de)

## Installation and requirements
A working `Python` 3.5 installation and the following libraries are required: 
`matplotlib`, `numpy`, `sklearn`, `scipy`, `ot` and `networkx`.
Having all dependencies available, `novoSpaRc` can be employed by cloning the 
repository, modifying the template `reconstruct_tissue.py` accordingly
and running it to perform the spatial reconstruction.

The code is partially based on adjustments of the POT (Python Optimal Transport) library (https://github.com/rflamary/POT).

`environments_and_versions.txt` contains environments in which we 
successfully tested `novoSpaRc`.


## General usage 
To spatially reconstruct gene expression, `novoSpaRc` performs the following
steps:
1. Read the gene expression matrix.

    1a. Optional: select a random set of cells for the reconstruction.
    
    1b. Optional: select a small set of genes (e.g. highly variable).

2. Construct the target space.

3. Setup the optimal transport reconstruction.

    3a. Optional: use existing information of marker genes, if available.

4. Perform the spatial reconstruction.

    4a. assigning cells a probability distribution over the target space.

    4b. derive a virtual in situ hybridization (vISH) for all genes over the target space.

5. Write outputs to file for further use, such as the spatial gene expression
matrix and the target space coordinates.

6. Optional: plot spatial gene expression patterns.

7. Optional: identify and plot spatial archetypes.

## Demonstration code
We provide scripts that spatially reconstruct two of the tissues presented
in the paper: the intestinal epithelium ([Moor, A.E., *et al*., Cell, 2018](https://www.sciencedirect.com/science/article/pii/S0092867418311644?via%3Dihub))
and the stage 6 Drosophila embryo ([Berkley Drosophila Transcription Network Project](http://bdtnp.lbl.gov)).

### The intestinal epithelium
The `reconstruct_intestine_denovo.py` script reconstructs the crypt-to-villus axis of the mammalian intestinal epithelium, based on data from Moor *et al*. 
The reconstruction is performed *de novo*, without using any marker genes. 
The script outputs plots of (a) a histogram showing the distribution of assignment values over embedded zones for each original villus zone, and (b) average spatial gene expression over the original villus zones and embedded zones of 4 gene groups.

Running time on a standard computer is under a minute.

### The *Drosophila* embryo
The `reconstruct_bdtnp_with_markers.py` script reconstructs the early
*Drosophila* embryo with only a handful of markers, based on the BDTNP dataset. 
All cells are used and
a random set of 1-4 markers is selected. The script outputs plots of
gene expression for a list of genes, as well as Pearson correlations of the
reconstructed and original expression values for all genes.
Notice that the results depend on which marker genes are selected. 
In the manuscript we averaged the results over many different choices of marker genes.

Running time on a standard desktop computer is around 6-7 minutes.

## Running novoSpaRc on your data
A template file for running `novoSpaRc` on custom datasets is 
provided (`reconstruct_tissue.py`). To successfully run `novoSpaRc` modify the
template file accordingly.

