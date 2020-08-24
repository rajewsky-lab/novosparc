from __future__ import print_function

###########
# imports #
###########

import numpy as np
from matplotlib.image import imread
import math
import random
#############
# functions #
#############

def construct_torus(num_cells):
    num_pts = num_cells
    indices = np.arange(0, int(np.sqrt(num_pts)), dtype=float) + 0.5

    angle = np.linspace(0, 2 * np.pi, int(np.sqrt(num_pts)))
    theta, phi = np.meshgrid(angle, indices)

    r, R = .25, 1.
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)

    locations = np.array(list(zip(x, y, z)))
    
    return locations

def construct_sphere(num_cells):
    num_pts = num_cells
    indices = np.arange(0, num_pts, dtype=float) + 0.5

    phi = np.arccos(1 - 2*indices/num_pts)
    theta = math.pi * (1 + 5**0.5) * indices

    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi);
    locations = np.array(list(zip(x, y, z)))
    
    return locations

def construct_circle(num_cells):
    num_pts = num_cells
    indices = np.arange(0, num_pts, dtype=float) + 0.5

    r = np.sqrt(indices/num_pts)
    theta = math.pi * (1 + 5**0.5) * indices

    x =  r*np.cos(theta)
    y =  r*np.sin(theta)
    
    locations = np.array(list(zip(x, y)))
    return locations

def construct_torus_2d(num_cells, radius=0.5):
    num_pts = num_cells
    indices = np.arange(0, num_pts, dtype=float) + 0.5

    r = np.sqrt(indices/num_pts)
    theta = math.pi * (1 + 5**0.5) * indices

    rs =[]
    thetas = []

    for ro, to in zip(r, theta):
        if ro > radius:
            rs.append(ro)
            thetas.append(to)

    x =  rs*np.cos(thetas)
    y =  rs*np.sin(thetas)
    
    locations = np.array(list(zip(x, y)))
    return locations

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
