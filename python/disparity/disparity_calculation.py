"""
The algorithm to compute the cost 
----
@version v1.1 - Januar 2017
@author Luca Palmieri
"""

import numpy as np
import scipy.ndimage as ndimage
import math
import plenopticIO.lens_grid as rtxhexgrid
import matplotlib.pyplot as plt
import pdb

def sweep_to_shift_costs(sweep_costs, max_cost):

    """
    Averages the lens cost to a single cost slice

    Parameters
    ----------

    sweep_costs: array-like
                 The costs of the plane sweep

    max_cost:    float
                 Maximum cost used for the sweeping

    Returns
    -------

    c: one-dimensional array
       Coarse cost slice for this lens
       
    """
    
    #pdb.set_trace()
    c = []
    
    for lens_costs in sweep_costs:
        ctmp = []
        for d in lens_costs:
            v = d[d < max_cost]
            # avoid division by zero by np.mean for empty v
            if v.shape[0] > 0:
                ctmp.append(np.mean(v))
            else:
                ctmp.append(max_cost)
                #ctmp.append(0)
                           
        c.append(ctmp)

    return np.array(c)

def lens_sweep(src_lens, dst_lenses, disparities, technique, hws=1, max_cost=10.0):


    """
    Returns the cost volume for a single lens plane sweep
    
    Parameters
    ----------
    
  

    hws: integer, optional
        half window size for the matching window

    max_cost: float, optional
        maximal cost 
      
    Returns
    -------

    res: array like, four dimensional
      The final cost volume. Axis 0: neighbour lens, Axis 1: disp, Axis 2: y, Axis 3: x
    src_img: array like, two-dimensional
      The source image
    disparities: array like, one dimensional, integer
      The disparities used

    """
    lens_grid = src_lens.grid
    disparities = np.asarray(disparities)
    src_img = src_lens.img_interp(lens_grid.y, lens_grid.x)

    # initialize the final cost volume
    cost = np.zeros((len(dst_lenses), len(disparities), src_img.shape[0], src_img.shape[1]))

    # number of filter elements (neighbourhood area)
    hws2 = (2*hws + 1)**2

    # visibility mask
    vis = np.zeros(src_img.shape)

    for i, dst_lens in enumerate(dst_lenses):

        # unit vector between the destination and source lens
        dv = dst_lens.pcoord - src_lens.pcoord
        dvn = np.linalg.norm(dv)
        dv /= dvn

        # lens distance in lens unit
        lens_dist = dvn / dst_lens.diameter
        
        dst_1d_orig = np.vstack((lens_grid.y, lens_grid.x)).T

        # the steps in the direction dv in pixels, adjust according to the lens distance
    
        for j, d in enumerate(disparities * lens_dist):
                
            # discard higher disparities, fill with last valid cost
            if d > 2*src_lens.inner_radius:
                cost[i, j:] = max_cost
                break

            # corresponding points in the target lens, 2d grids
            dst_y = lens_grid.yy - d*dv[0]
            dst_x = lens_grid.xx - d*dv[1]


            # corresponding points in the target lens, 1d axis
            dst_1d = dst_1d_orig - d * dv
            
            # mask inner radius of the source and the target lens
            mask_ind = ((dst_y**2 + dst_x**2) > src_lens.inner_radius**2)
            mask_ind_inv = (mask_ind < 1)

            # retrieve the interpolated destination patch

            dst_img = dst_lens.img_interp(dst_1d[:, 0], dst_1d[:, 1])    
            
            # Here the costs are calculated using the chosen technique
            if technique == 'ssd':
                
                diff = np.pow((dst_img - src_img), 2)
                
            elif technique == 'census':
            
                w, h = src_img.shape

                #Initialize output array
                census_src = np.zeros((h-2, w-2), dtype='uint8')
                census_dst = np.zeros((h-2, w-2), dtype='uint8')
                
                #centre pixels, which are offset by (1, 1)
                cp = src_img[1:h-1, 1:w-1]

                #offsets of non-central pixels 
                offsets = [(u, v) for v in range(3) for u in range(3) if not u == 1 == v]

                #Do the pixel comparisons
                for u,v in offsets:
                    census_src = (census_src << 1) | (src_img[v:v+h-2, u:u+w-2] >= cp)   
                    census_dst = (census_dst << 1) | (dst_img[v:v+h-2, u:u+w-2] >= cp)

                #Convert transformed data to image
                src_cens_img = np.zeros(src_img.shape)
                src_cens_img[1:h-1,1:w-1] = census_src
                dst_cens_img = np.zeros(src_img.shape)
                dst_cens_img[1:h-1,1:w-1] = census_dst
                diff = np.abs(dst_cens_img - src_cens_img)
                diff /= 255.0
            
            elif technique == 'ncc':
                
                ncc = calculate_ncc(src_img, dst_img) 
                ncc += 1
                ncc /= 2
                diff = 1 - ncc            
 
            else:
                
                diff = np.abs(dst_img - src_img)

            # select only visible area of the costs
            diff *= mask_ind_inv
            # simple box filter of size (2*hws+1)**2
            ndimage.uniform_filter(diff, output=cost[i, j], size=2*hws+1)
            # uniform filter divides by the size, invert this process
            cost[i, j] *= hws2
            # get the number of visible pixels in the local area
            ndimage.uniform_filter(hws2 * mask_ind_inv, output=vis, size=2*hws+1)
           
            # mask the the averaged pixels
            vis *= mask_ind_inv
            vis_mask = vis > 0
            
            # divide cost by visible pixels
            cost[i, j][vis_mask] /= vis[vis_mask]

            cost[i, j][mask_ind] = max_cost

            cost[i, j][src_lens.mask < 1] = max_cost

    return cost, src_img, disparities

#@jit
def calculate_ncc(src_img, dst_img):
    
    diff = np.zeros((src_img.shape))
    d = 1
    for k in range(d, src_img.shape[0] - (d + 1)):
        for l in range(d, src_img.shape[1] - (d + 1)):
            diff[k, l] = correlation_coefficient(dst_img[k - d: k + d + 1, l - d: l + d + 1], src_img[k - d: k + d + 1, l - d: l + d + 1])
    return diff 
    
#@jit   
def correlation_coefficient(patch1, patch2):
    product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
    stds = patch1.std() * patch2.std()
    if stds == 0:
        return 0
    else:
        product /= stds
        return product

def convertRGB2Gray(img):

    img_gray = np.zeros(shape=(img.shape[0], img.shape[1]))
    
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            img_gray[x,y] = 0.299 * img[x,y,0] + 0.587 * img[x,y,1] + 0.114 * img[x,y,2]

    return img_gray

def merge_costs_additive(cost_volumes, max_cost):
    
    """
    Additive sum of the costs, normalization according to the number
    of lenses who potentially see point

    Parameters
    ----------

    cost_volumes: list of arrays
      List of cost volumes (one volume for each lens)

    max_cost: float
      Maximal cost used in the sweeping process     
    

    Returns
    -------

    merged_cost: array-like
      Single merged cost volume

    valid_sum: array-like, same size as merged_cost
      Visibility volume
    """
    
    l = cost_volumes.shape[0]
    
    # mask: 1 for visible pixels, 0 else
    valid = cost_volumes < max_cost

    # how many lenses are able to see a pixel? Sum the mask along the
    # cost axis
    valid_sum = valid.sum(axis=0)
    
    # sum the visible costs
    merged_cost = (cost_volumes * valid).sum(axis=0)

    # select the pixels which are visible to at least one lens
    ind = valid_sum > 0

    # normalize the costs
    merged_cost[ind] /= (1.0 * valid_sum[ind])
    
    # cost for completely invisible pixels
    merged_cost[valid_sum < 1] = max_cost * l

    return merged_cost


def cost_minima_interp(cost_volume, x, dx=None):

    """
    Parameters
    ----------

    cost_volume: array_like, shape (h, w, d)
    x:           array_like, shape (d, )
    dx:          double
                 step size in x

    Returns
    ---------

    min_interp:  double
                 Interpolated minima using a Taylor series expansion

    val_interp:  double
                 Interpolated function values at min_interp

    """

    if dx is None:
        # assumption: values in x are equidistant and positive, increasing
        assert len(x) > 2
        dx = x[1] - x[0]
    
    # indices of the plain minima
    min_plain = np.argmin(cost_volume, axis=2)

    # first order derivatives along the third axis
    d1 = 1.0 * np.gradient(cost_volume, dx)[2]
    
    # second order derivatives along the third axis
    d2 = 1.0 * np.gradient(d1, dx)[2]

    # select minima with second derivative > 0
    ind = np.where((min_plain > 0) * (min_plain < (len(x) - 1)))
    tmp_ind_depth = (ind[0], ind[1], min_plain[ind])
    tmp_ind = np.where(d2[tmp_ind_depth] > 0)
    ind = (ind[0][tmp_ind], ind[1][tmp_ind])
    ind_depth = (ind[0], ind[1], min_plain[ind])


    min_interp = 1.0 * np.zeros_like(min_plain)
    
    # interpolation using a second order Taylor series expansion
    min_interp[ind] = x[min_plain[ind]] - (d1[ind_depth] / d2[ind_depth])

    xtmp = (x[min_plain[ind]] - min_interp[ind])

    # calculate the function values at the interpolated points
    val_interp = np.zeros_like(min_interp)
    val_interp[ind] = cost_volume[ind_depth] + d1[ind_depth] * xtmp + 0.5 * (d2[ind_depth] * xtmp**2)

    # TODO: Experimental: set the values for minima at index 0 and last
    ind = np.where((min_plain == 0) + (min_plain == (len(x) - 1)))
    #print(min_plain[ind])
    min_interp[ind] = x[min_plain[ind]]
    ind_depth = (ind[0], ind[1], min_plain[ind])
    val_interp[ind] = cost_volume[ind_depth]
    
    
    return min_interp, val_interp

def cost_minimum_interp(cost_slice, x, dx=None):

    #TODO: all cases ?
    
    """
    Parameters
    ----------

    cost_slice: array_like, shape (d, )
    x:          array_like, shape (d, )
    dx:         double
                step_size in x

    Returns
    ---------

    min_interp: double,
                Interpolated minima using a Taylor series expansion

    val_interp: double,
                Interpolated function value at min_interp
    """
     
    if dx is None:
        assert len(x) > 2
        dx = x[1] - x[0]

    # index of the plain minimum
    min_plain = np.argmin(cost_slice)

    # no interpolation possible in these cases, return the plain values
    if min_plain == 0 or min_plain == len(x)-1:
        return x[min_plain], cost_slice[min_plain]
    
    # first order derivative
    d1 = 1.0 * np.gradient(cost_slice, dx)
    
    # second order derivative
    d2 = 1.0 * np.gradient(d1, dx)

    min_interp = x[min_plain]
    
    if d2[min_plain] > 0:
        # interpolation using a second order Taylor series expansion
        min_interp = min_interp - (d1[min_plain] / d2[min_plain])

    xtmp = x[min_plain] - min_interp
    val_interp = cost_slice[min_plain] + d1[min_plain] * xtmp + 0.5 * (d2[min_plain] * xtmp**2)
    
    return min_interp, val_interp

def assign_last_valid(cost_volume, max_cost=None, axis=0):

    
    if max_cost is None:
        max_cost = np.amax(cost_volume)
        #print("max cost {0}".format(max_cost))

    ind1 = np.argmax(cost_volume >= max_cost, axis=axis)
    ind2 = np.where(ind1 > 0)

    tmp = cost_volume[:, ind2[0], ind2[1]]

    ind3 = np.where(cost_volume[:, ind2[0], ind2[1]] >= max_cost)
    tmp[ind3] = cost_volume[ind1[ind2] - 1, ind2[0], ind2[1]][ind3[1]]
    cost_volume[:, ind2[0], ind2[1]] = tmp

    return

