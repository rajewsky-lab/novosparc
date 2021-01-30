import pandas as pd
import numpy as np
import anndata as ad
import scanpy as sc

def load_data(path, dtype='dge'):
    if dtype == 'dge':
        dataset = ad.read_text(path)

    elif dtype == '10x':
        dataset = sc.read_10x_mtx(path,  var_names='gene_symbols',  cache=True)

    return dataset


def load_target_space(path, cells_selected=None, coords_cols=None):
    locations = pd.read_csv(path, sep='\t')

    if coords_cols:
        locations = locations[coords_cols]

    if cells_selected:
        locations = locations.iloc[cells_selected, :]

    return locations.values
