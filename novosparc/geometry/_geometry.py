from __future__ import print_function

###########
# imports #
###########

import numpy as np
from matplotlib.image import imread

#############
# functions #
#############

def construct_target_grid(num_cells):
    """Constructs a rectangular grid. First a grid resolution is randomly
    chosen. grid_resolution equal to 1 implies equal number of cells and
    locations on the grid. The random parameter beta controls how rectangular
    the grid will be -- beta=1 constructs a square rectangle.
    num_cells -- the number of cells in the single-cell data."""

    grid_resolution = int(np.random.randint(1, 2+(num_cells/1000), 1))
    grid_resolution = 2
    num_locations = len(range(0, num_cells, grid_resolution))
    grid_dim = int(np.ceil(np.sqrt(num_locations)))

    beta = round(np.random.uniform(1, 1.5), 1) # controls how rectangular the grid is
    # beta = 1 # set this for a square grid
    x = np.arange(grid_dim * beta)
    y = np.arange(grid_dim / beta)
    locations = np.array([(i, j) for i in x for j in y])

    return locations

def create_target_space_from_image(image):
    """Create a tissue target space from a given image. The image is assumed to
    contain a black-colored tissue space in white background.
    image -- the location of the image on the disk."""
    img = imread(image)
    img_width = img.shape[1]
    img_height = img.shape[0]

    locations = np.array([(x, y) for x in range(img_width) for y in range(img_height)
                          if sum(img[y, x, :] == np.array([0, 0, 0]))])

    return locations
