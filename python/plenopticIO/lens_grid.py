"""
It contains the information about the hexagonal grid of the micro-lenses
----
@version v1 - December 2017
@author Luca Palmieri
"""

import numpy as np
import pdb
import matplotlib.pyplot as plt

# coefficients of the basis vectors (pby, pbx) in lens units for the
# first three rings of the same lens type

# type 1: current
# type 2: right neighbour
# type 3: left neighbour

HEX_TYPE_OFFSETS = [
                    [
                     [[ 1,  1], [ 2, -1], [ 1, -2],
                      [-1, -1], [-2,  1], [-1,  2]],
                     [[0,  3], [ 3, 0], [ 3, -3], 
                      [0, -3], [-3, 0], [-3,  3]]
                    ],
                    [
                     [[0, 1], [1, -1], [-1, 0]],
                     [[2, 0], [0, -2], [-2, 2]],
                     [[ 1,  2], [ 3, -2], [ 2, -3],
                      [-2, -1], [-3,  1], [-1,  3]]
                    ],
                    [
                     [[1, 0], [0, -1], [-1, 1]],
                     [[0, 2], [2, -2], [-2, 0]],
                     [[ 2,  1], [ 3, -1], [ 1, -3],
                      [-1, -2], [-3,  2], [-2,  3]]
                    ]
                  ]

# neighbours of all types, sorted by distance in lens units
# ring[0]: 1 lens
# ring[1]: 1 * sqrt*(3) lenses
# ring[2]: 2 lenses
# ring[3]: ...

HEX_OFFSETS = [
                 [
                  [ 0,  1], [ 1,  0], [ 1, -1],
                  [ 0, -1], [-1,  0], [-1,  1]
                 ],
                 [
                  [ 1,  1], [ 2, -1], [ 1, -2],
                  [-1, -1], [-2,  1], [-1,  2]
                 ],
                 [
                  [ 0,  2], [ 2,  0], [ 2, -2], 
                  [ 0, -2], [-2,  0], [-2,  2]
                 ],
                 [
                  [ 1,  2], [ 2,  1], [ 3, -1], [ 3, -2],
                  [ 2, -3], [ 1, -3], [-1, -2], [-2, -1],
                  [-3,  1], [-3,  2], [-2,  3], [-1,  3]
                 ],
                 [
                  [0,  3], [ 3, 0], [ 3, -3],
                  [0, -3], [-3, 0], [-3,  3]
                 ],
                 [
                  [ 2,  2], [ 4, -2], [ 2, -4],
                  [-2, -2], [-4,  2], [-2,  4]
                 ],
                 [
                  [ 1,  3], [ 3,  1],  [ 4, -1], [ 4, -3],
                  [ 3, -4], [ 1, -4],  [-1, -3], [-3, -1],
                  [-4,  1], [-4,  3],  [-3,  4], [-1,  4]
                 ],
                 [
                  [0,  4], [ 4, 0], [ 4, -4],
                  [0, -4], [-4, 0], [-4,  4]
                 ]
               ]

HEX_DIRECTIONS = HEX_OFFSETS[0]

SCAM_OFFSETS = [[2,-2], [1,-1], [2,-1], [1,0], [2,0], [0,-2], [0,-1], [0,1], [0,2], [-2,0], [-1,0], [-2,1],[-1,1],[-2,2]]

class HexDir(object):
    E =  HEX_DIRECTIONS[0]
    SE = HEX_DIRECTIONS[1]
    SW = HEX_DIRECTIONS[2]
    W =  HEX_DIRECTIONS[3]
    NW = HEX_DIRECTIONS[4]
    NE = HEX_DIRECTIONS[5]
    
def hex_focal_type(c):
    
    """
    Calculates the focal type for the three lens hexagonal grid
    """

    focal_type = ((-c[0] % 3) + c[1]) % 3

    return focal_type 
    

def hex_lens_grid(img_shape, diam, angle, offset, B, filter_method='lens'):
    
    """
    Parameters:
    
    h: integer
       image height in pixels
    w: integer
        image width in pixels
    diam: float
        lens diameter in pixels
    angle: float
        rotation angle of the grid, counter-clockwise in radians
    offset: array-like
        (y, x) offset in pixels of the center lens
    B: array-like
        2x2 matrix, basis vectors of the hex grid, each colum represents a basis vector
    
    filter_method: string
        'lens': removes all lenses which are not completely inside the image region
        'center': removes all lenses whose centers are not within the image region
        
    Returns: centers, array-like
             array of (y, x) lens center coordinates in pixels
             
    Convention: Origin of the image: (0, 0) upper left corner, positive y axis points down
    """

    B = np.asarray(B)
   

    if np.linalg.norm(B[0]) > 1:
        switch_xy = True
    else:
        switch_xy = False

    #print("Basis: {0} switch {1}".format(B, switch_xy))
    
    lens_centers = []
    h, w = img_shape
    img_center = np.array([(h-1)/2.0, (w-1)/2.0])
    
    # conservative estimate of lenses in y and x direction to cover the image
    ny = int(np.ceil(h * np.sqrt(2) / diam) + 2)
    nx = int(np.ceil(w * np.sqrt(2) / diam) + 4)
    
    r = diam / 2.0
    
    # start offsets, upper left corner
    sx = - (nx * diam - w) / 2.0
    sy = - (ny * r * np.sqrt(3) - h) / 2.0
    
    # generate the lens center coordinates in pixels
    for i in range(ny):
        py = i * np.sqrt(3) * r + sy
        for j in range(nx):
            px = (sx + j * diam) + (i % 2) * r
            
            if switch_xy is True:
                lens_centers.append(np.array([px, py]))
            else:
                lens_centers.append(np.array([py, px]))

    # lens center closest to the image center
    lorigin = _lens_origin(lens_centers, img_center)

    # adjust the grid: lorigin is the new origin
    lens_centers = [c - lorigin for c in lens_centers]
    
    lenses = _axial_coordinates(B * diam, lens_centers)
    tlenses = _transform_grid(lenses, img_center, angle, offset)
    
    filters = dict()
    filters['center'] = lambda p: p[0] >=0 and p[0] < h and p[1] >= 0 and p[1] < w
    filters['lens']   = lambda p: (p[0] - r) >=0 and (p[0] + r) < h and  \
                                  (p[1] - r) >= 0 and (p[1] + r) < w
    
    if filter_method in filters:
        
        tlenses = {key: tlenses[key] for key in tlenses 
                   if filters[filter_method](tlenses[key])}
        
    return tlenses


def hex_lens_grid_plus(img_shape, diam, angle, offset, B, filter_method='lens'):
    
    """
    Parameters:
    
    h: integer
       image height in pixels
    w: integer
        image width in pixels
    diam: float
        lens diameter in pixels
    angle: float
        rotation angle of the grid, counter-clockwise in radians
    offset: array-like
        (y, x) offset in pixels of the center lens
    B: array-like
        2x2 matrix, basis vectors of the hex grid, each colum represents a basis vector
    
    filter_method: string
        'lens': removes all lenses which are not completely inside the image region
        'center': removes all lenses whose centers are not within the image region
        
    Returns: centers, array-like
             array of (y, x) lens center coordinates in pixels
             
    Convention: Origin of the image: (0, 0) upper left corner, positive y axis points down
    """

    B = np.asarray(B)
   

    if np.linalg.norm(B[0]) > 1:
        switch_xy = True
    else:
        switch_xy = False

    #print("Basis: {0} switch {1}".format(B, switch_xy))
    
    lens_centers = []
    h, w = img_shape
    img_center = np.array([(h-1)/2.0, (w-1)/2.0])
    
    # conservative estimate of lenses in y and x direction to cover the image
    ny = int(np.ceil(h * np.sqrt(2) / diam) + 2)
    nx = int(np.ceil(w * np.sqrt(2) / diam) + 4)
    
    r = diam / 2.0
    
    # start offsets, upper left corner
    sx = - (nx * diam - w) / 2.0
    sy = - (ny * r * np.sqrt(3) - h) / 2.0
    
    # generate the lens center coordinates in pixels
    for i in range(ny):
        py = i * np.sqrt(3) * r + sy
        for j in range(nx):
            px = (sx + j * diam) + (i % 2) * r
            
            if switch_xy is True:
                lens_centers.append(np.array([px, py]))
            else:
                lens_centers.append(np.array([py, px]))

    # lens center closest to the image center
    lorigin = _lens_origin(lens_centers, img_center)

    # adjust the grid: lorigin is the new origin
    lens_centers = [c - lorigin for c in lens_centers]
    
    lenses = _axial_coordinates(B * diam, lens_centers)
    tlenses = _transform_grid(lenses, img_center, angle, offset)
    

    filters = dict()
    filters['center'] = lambda p: p[0] >=0 and p[0] < h and p[1] >= 0 and p[1] < w
    filters['lens']   = lambda p: (p[0] - r) >=0 and (p[0] + r) < h and  \
                                  (p[1] - r) >= 0 and (p[1] + r) < w
    
    if filter_method in filters:
        
        tlenses = {key: tlenses[key] for key in tlenses 
                   if filters[filter_method](tlenses[key])}

    return tlenses, ny, nx, sy, sx, img_center
   
def _lens_origin(lens_centers, img_center):
    
    
    """
    Parameters:
    
    h: integer, image height in pixels
    w: integer, image width in pixels
    centers: array of lens center coordinates (y, x) in pixels
    
    Returns: 
    
    centers: array-like, lens center closest to the image center
    """
    
    # distance from the lens centers to the image center
    dist = [np.linalg.norm(c - img_center) for c in lens_centers]
    return lens_centers[np.argmin(dist)]

def _axial_coordinates(B, centers):
    
    """
    Parameters:
    
    B: 2x2 matrix, grid basis vectors
    
    Returns: 
    
    axial_coords: dictionary, keys: axial coordinates, 
                  values:center coordinates in pixels
    """
    
    lenses = dict()

    for c in centers:
        tmp = np.linalg.solve(B, c)
        axial_coord = tuple(np.round(tmp).astype(int))
        lenses[axial_coord] = c

    return lenses

def _transform_grid(lenses, img_center, angle, offset):
    
    # rotation matrix, counter-clockwise
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    
    acenters = {key: np.dot(R, lenses[key]) + offset + img_center
                for key in lenses}
        
    return acenters
