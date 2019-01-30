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
A working `Python 2.7/3.3` installation and the following libraries are required: 
`matplotlib`, `numpy`, `sklearn`, `scipy`, `ot` and `networkx`.
Having all dependencies available, novoSpaRc can be employed by cloning the 
repository, modifying the template `reconstruct_tissue.py` accordingly
and running it to perform the spatial reconstruction.

novoSpaRc has been successfully tested in Ubuntu 16.04 with `Python 3.5`
and the following library versions: `matplotlib` v2.2.2, `numpy` v1.14.2,
`sklearn` v0.19.1, `scipy` v1.0.0, `ot` v0.4.0, `networkx` v2.0
and Mac OS X 10.7.

## General usage 
To spatially reconstruct gene expression, novoSpaRc performs the following
steps:
1. Read the gene expression matrix.

    1a. Optional: select a random set of cells for the reconstruction.

    1b. Optional: subset to a small set of genes (highly variable or other).
2. Construct the target space.
3. Setup the optimal transport reconstruction.

    3a. Optional: if existing information of marker genes is available, use it.
4. Perform the spatial reconstruction.
5. Write outputs to file for further use, such as the spatial gene expression
matrix and the target space coordinates.
6. Optional: plot spatial gene expression patterns.
7. Optional: identify spatial archetypes. 

## Demonstration code
We provide scripts that spatially reconstruct two of the tissues presented
in the paper: the intestinal epithelium [REF] and the stage 6 Drosophila 
embryo ([Berkley Drosophila Transcription Network Project](http://bdtnp.lbl.gov)).

### The intestinal epithelium

### The *Drosophila* embryo
The `reconstruct_bdtnp_with_markers.py` script reconstructs the early
*Drosophila* embryo with only a handful of markers. All cells are used and
a random set of 1-4 markers is selected. The script outputs plots of
gene expression for a list of genes, as well as Pearson correlations of the
reconstructed and original expression values for all genes.
As the results depend on which 
marker genes are selected, note that the output might slightly differ 
from the one found here on the repository. Running time on a desktop 
computer with two cores is around 6-7 minutes.

## Running novoSpaRc on your data
A template file for running novoSpaRc on custom datasets is 
provided (`reconstruct_tissue.py`). To successfully run novoSpaRc modify the
template file accordingly.

