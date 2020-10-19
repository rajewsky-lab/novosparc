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

def construct_torus(num_locations):
    num_pts = num_locations
    indices = np.arange(0, int(np.sqrt(num_pts)), dtype=float) + 0.5

    angle = np.linspace(0, 2 * np.pi, int(np.sqrt(num_pts)))
    indices = np.arange(0, int(np.sqrt(num_pts)), dtype=float) + 0.5

    angle = np.linspace(0, 2 * np.pi, int(np.sqrt(num_pts)))
    theta, phi = np.meshgrid(angle, indices)

    r, R = .25, 1.
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)

    locations = np.array(list(zip(x, y, z)))
    
    return locations

def construct_sphere(num_locations):
    num_pts = num_locations
    indices = np.arange(0, num_pts, dtype=float) + 0.5

    phi = np.arccos(1 - 2*indices/num_pts)
    indices = np.arange(0, num_pts, dtype=float) + 0.5

    phi = np.arccos(1 - 2*indices/num_pts)
    theta = math.pi * (1 + 5**0.5) * indices

    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi);
    locations = np.array(list(zip(x, y, z)))
    
    return locations

def construct_circle(num_locations, random=False):
    num_pts = num_locations

    if random:
        theta = np.linspace(0, 2 * np.pi, num_pts)
        r = np.random.rand((num_pts))
    else:
        indices = np.arange(0, num_pts, dtype=float) + 0.5
        r = np.sqrt(indices / num_pts)
        theta = math.pi * (1 + 5 ** 0.5) * indices

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    locations = np.array(list(zip(x, y)))
    return locations

def construct_torus_2d(num_locations, radius=0.5, random=False):
    num_pts = num_locations
    indices = np.arange(0, num_pts, dtype=float) + 0.5

    if random:
        r = np.random.rand((num_pts))
    else:
        r = np.sqrt(indices / num_pts)
    theta = math.pi * (1 + 5 ** 0.5) * indices

    rs = []
    thetas = []

    for ro, to in zip(r, theta):
        if ro > radius:
            rs.append(ro)
            thetas.append(to)

    x = rs * np.cos(thetas)
    y = rs * np.sin(thetas)

    locations = np.array(list(zip(x, y)))
    return locations

def construct_target_grid(num_locations, ratio=1.2, random=False):
    grid_dim = int(np.ceil(np.sqrt(num_locations / ratio)))

    if random:
        grid_dim = int(np.ceil(np.sqrt(num_locations * 2 / ratio)))
        x = np.arange(grid_dim * ratio)
        y = np.arange(grid_dim)
        locations = np.array([(i, j) for i in x for j in y])
        locations = locations[np.random.choice(np.arange(len(locations)), num_locations, replace=False)]
    else:
        x = np.arange(grid_dim * ratio)
        y = np.arange(grid_dim)
        locations = np.array([(i, j) for i in x for j in y])

    return locations

def construct_line(num_locations):
    return np.vstack((range(num_locations), np.ones(num_locations))).T

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
