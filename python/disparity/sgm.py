
import numpy as np
import raytrix.sgm_cost_path as scp
# semi global matching
import pdb

__all__ = ['sgm', 'consistency_check', 'uniqueness_check']


def _gen_starting_points(h, w):

    # the starting points are the border pixels


    # in python 3, zip returns a builtin, convert to list 
    s0 = list(zip(range(h), [0]*w))
    s1 = list(zip(range(h), [w-1]*h))
    s2 = list(zip([0]*w, range(w)))
    s3 = list(zip([h-1]*w, range(w)))


    s4 = s0 + s2
    s5 = s1 + s2
    s6 = s3 + s0
    s7 = s1 + s3

    return [s0, s1, s2, s3, s4, s5, s6, s7]

def _filter_starting_points(starting_points, directions, mask):
    
    # restrict the cost paths along *directions* to the mask

    h, w = mask.shape
    filtered_starting_points = []

    for i, d in enumerate(directions):
        tmp_points = []
        for p in starting_points[i]:
            
            while p[0] >= 0 and p[0] < h and p[1] >= 0 and p[1] < w:
                
                if mask[p[0], p[1]]:
                    tmp_points.append(p)
                    break
                else:
                    p += d

        filtered_starting_points.append(np.array(tmp_points))

    return filtered_starting_points
            
            
#@profile
#@jit(double[:, :, :](double[:, :], double[:, :, :], double[:, :], double, double, boolean, double))
#@autojit
def sgm(ref_img, cost_volume, mask=None, penalty1=0.1, penalty2=0.5, only_dp=False, max_cost=10.0):

    """
    Semi-Global-Matching Cost-volume filtering as proposed by Heiko Hirschmueller
    
    
    Parameters
    ----------
    
    ref_img: array_like, shape (h, w)
             The reference intensity image

    cost_volume: array_like, shape (h, w, l)
             Cost volume, e.g. matching costs for a set of disparities
    
    mask: array_like, shape (h, w)
             Mask for path restriction (non-rectangular domains)

    penalty1: float, optional
             Penalty for small disparity differences (+- 1)
  
    penalty2: float, optional
             Penalty for disparity differences > 1

    only_dp: boolean, optional
             Use only the classical dynamic programming stereo variant (one direction)

    
    Returns
    -------

    accum_volume: array_like, shape (h, w, l)
             The accumulated cost volume
    
    """

    h, w, _ = cost_volume.shape
    assert (h, w) == ref_img.shape

    
    accum_volume = np.zeros_like(cost_volume)
    
    all_directions = np.array([[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]])
    all_starting_points = _gen_starting_points(h, w)
    
    if only_dp:
        # use only one direction for the regularization (right to left)
        # this corresponds to the classical scanline dynamic programming variant
        directions = all_directions[1:2]
        starting_points = all_starting_points[1:2]
    else:
        # use all 8 directions, this corresponds the the Hirschmueller variant
        directions = all_directions
        starting_points = all_starting_points
    
    if mask is None:
        mask = np.ones_like(ref_img)
    else:
        assert np.all(ref_img.shape == mask.shape)
        starting_points = _filter_starting_points(starting_points, directions, mask)

    for i, d in enumerate(directions):
        
        # temporary cost volume for the current direction
        tmp_accum_volume = np.zeros_like(cost_volume)

        for p in starting_points[i]:
            #tmp_accum_volume += _cost_path(np.array(p), ref_img, cost_volume, d, mask=mask, penalty1=penalty1, penalty2=penalty2, max_cost=max_cost)
            #_cost_path(np.array(p), ref_img, cost_volume, tmp_accum_volume, d, mask=mask, penalty1=penalty1, penalty2=penalty2, max_cost=max_cost)
            scp.cost_path(np.array(p), ref_img, cost_volume, tmp_accum_volume, d, mask=mask, penalty1=penalty1, penalty2=penalty2, max_cost=max_cost)
            # for the autojit variant
            #tmp_accum_volume += _cost_path(np.array(p), ref_img, cost_volume, d, mask, penalty1, penalty2, max_cost)
            
        accum_volume += tmp_accum_volume
            
    return accum_volume

#@profile
#@autojit
def _cost_path(p, ref_img, cost_volume, accum_volume, direction, mask=None, penalty1=0.1, penalty2=0.2, max_cost=1.0):
    
    #TODO: uniqueness check: set to invalid if minimum is not unique

    ptmp = np.copy(p)

    h, w, depth = cost_volume.shape 
    last_slice = accum_volume[ptmp[0], ptmp[1]]
    last_min = np.min(last_slice)
    last_intensity = ref_img[ptmp[0], ptmp[1]]
    ptmp += direction

    while ptmp[0] >= 0 and ptmp[0] < h and ptmp[1] >= 0 and ptmp[1] < w:

        if not mask[ptmp[0], ptmp[1]]:
            break
        
        cur_slice = accum_volume[ptmp[0], ptmp[1]]
        cur_intensity = ref_img[ptmp[0], ptmp[1]]

        # intensity gradient along the current path in the source image
        cur_gradient = np.abs(cur_intensity - last_intensity)

        # cost for disparities larger than 1px
        if cur_gradient > 0:
            penalty2_cost = last_min + penalty2 / cur_gradient
        else:
            penalty2_cost = last_min + penalty2

        # current slice minimum, initialize with max imum double
        cur_min = np.finfo('d').max
        
        for d in range(depth):
        
            # cost for equal disparity 
            eq_cost = last_slice[d]
            
            # cost for +- 1 disparity step
            if d > 0:
                penalty1m_cost = last_slice[d-1] + penalty1
            else:
                penalty1m_cost = 2 * max_cost
            if d < depth - 1:
                penalty1p_cost = last_slice[d+1] + penalty1
            else:
                penalty1p_cost = 2 * max_cost

            # accumulate
            cur_slice[d] = cost_volume[ptmp[0], ptmp[1], d] + min([eq_cost, penalty1m_cost, penalty1p_cost, penalty2_cost]) - last_min

            # update current min
            if cur_slice[d] < cur_min:
                cur_min = cur_slice[d]

        # update state variables for the next iteration
        last_min = cur_min
  
        last_slice = cur_slice
        last_intensity = cur_intensity
        ptmp += direction

  

def consistency_check(left_disparity, right_disparity, invalid=-1.0, max_difference=1):
    
    """ 
    
    Parameters
    ----------

    left_disparity: two-dimensional array, shape (h, w)
                    The left disparity image (reference image)

    right_disparity: two-dimensional array, shape (h, w)
                    The right disparity image

    invalid: float, optional
                    Value for inconsistent pixels

    max_difference: integer, optional
                    Maximal disparity difference
    

    Returns
    -------

    final_disparity: two-dimensional array, shape (h, w) 

                     Disparity image of the left viewpoint containing
                     the disparities from *left_disparities* for
                     consistent disparities and *invalid* for
                     inconsistent disparities
    """

                     
    assert np.all(left_disparity.shape == right_disparity.shape) 

    h, w = right_disparity.shape    
    final_disparity = np.ones_like(left_disparity) * invalid
   
    for i in range(h):
        for j in range(w):
            
            d1 = left_disparity[i, j] 
            
            if j-d1 < 0:
                continue

            d2 = right_disparity[i, j-d1]

            if abs(d1 - d2) <= max_difference:
                final_disparity[i, j] = d1
                

    return final_disparity

def uniqueness_check(cost_volume):
    pass
            
