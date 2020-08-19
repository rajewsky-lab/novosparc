import numpy as np
import anndata as ad


def load_data(path, dtype='dge'):
    if dtype == 'dge':
        dataset = ad.read_text(path)

    return dataset


def load_target_space(path, cells_selected, is_2D=True):
    locations = np.loadtxt(path, skiprows=1)
    if is_2D:
        locations = locations[:, [0, 2]]
    locations = locations[cells_selected, :]

    return locations
