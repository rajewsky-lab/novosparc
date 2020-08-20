import numpy as np
import anndata as ad
import scanpy as sc

def load_data(path, dtype='dge'):
    if dtype == 'dge':
        dataset = ad.read_text(path)

    elif dtype == '10x':
        dataset = sc.read_10x_mtx(path,  var_names='gene_symbols',  cache=True)

    return dataset


def load_target_space(path, cells_selected=None, is_2D=True):
    locations = np.loadtxt(path, skiprows=1)
    if is_2D:
        locations = locations[:, [0, 2]]

    if cells_selected is not None:
        locations = locations[cells_selected, :]

    return locations
