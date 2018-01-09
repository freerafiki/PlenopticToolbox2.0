import numpy as np
cimport numpy as np

#from libc.math import round, fabs
cimport libc.math as lm

DTYPE = np.double
ctypedef np.double_t DTYPE_t


def cost_path(p, double [:, :] ref_img, double[:, :, :]cost_volume, 
               double[:, :, :] accum_volume, direction, 
               double [:, :] mask, double penalty1=0.1, double penalty2=0.2, double max_cost=1.0):

    
    #TODO: uniqueness check: set to invalid if minimum is not unique

    cdef double dy = direction[0]
    cdef double dx = direction[1]

    cdef double pyd = p[0] 
    cdef double pxd = p[1] 
    
    cdef unsigned int py = <unsigned int>lm.round(pyd)
    cdef unsigned int px = <unsigned int>lm.round(pxd)

    cdef unsigned int lpy = py
    cdef unsigned int lpx = px

    cdef unsigned int h = cost_volume.shape[0]
    cdef unsigned int w = cost_volume.shape[1]
    cdef unsigned int depth = cost_volume.shape[2]
    cdef unsigned int d = 0
    
    
    #last_slice = accum_volume[ptmp[0], ptmp[1]]
    cdef double last_min = np.min(accum_volume[py, px])
    cdef double last_intensity = ref_img[py, px]
    cdef double cur_gradient = 0.0

    cdef double cur_intensity = 0.0
    cdef double maxconst = np.finfo('d').max
    cdef double tmp_cost = 0.0

    cdef double atmp = 0.0
    cdef double cur_min = 0.0

    cdef double eq_cost = 0.0
    cdef double penalty1p_cost = 0.0
    cdef double penalty1m_cost = 0.0
    cdef double penalty2_cost = 0.0

    cdef double penalty

    pyd += dy
    pxd += dx

    py = <unsigned int>lm.round(pyd)
    px = <unsigned int>lm.round(pxd)

    #ptmp += direction

    while py >= 0 and py < h and px >= 0 and px < w:

        if not mask[py, px]:
            break
        
        #cur_slice = accum_volume[ptmp[0], ptmp[1]]
        cur_intensity = ref_img[py, px]

        # intensity gradient along the current path in the source image
        cur_gradient = lm.fabs(cur_intensity - last_intensity)

        # cost for disparities larger than 1px
        if cur_gradient > 0:
            penalty2_cost = last_min + penalty2 / cur_gradient
        else:
            penalty2_cost = last_min + penalty2

        # current slice minimum, initialize with maximum double
        cur_min = maxconst
        
        d = 0

        #for d in range(depth):
        while d < depth:

            # cost for equal disparity 
            eq_cost = accum_volume[lpy, lpx, d]
            tmp_cost = eq_cost

            # cost for +- 1 disparity step
            if d > 0:
                penalty1m_cost = accum_volume[lpy, lpx, d-1] + penalty1
            else:
                penalty1m_cost = 2 * max_cost
            
            if penalty1m_cost < tmp_cost:
                tmp_cost = penalty1m_cost

            if d < depth - 1:
                penalty1p_cost = accum_volume[lpy, lpx, d+1] + penalty1
            else:
                penalty1p_cost = 2 * max_cost
                
            if penalty1p_cost < tmp_cost:
                tmp_cost = penalty1p_cost

            if penalty2_cost < tmp_cost:
                tmp_cost = penalty2_cost

            # accumulate
            #accum_volume[py, px, d] = 
            #a = cost_volume[py, px, d] + min([eq_cost, penalty1m_cost, penalty1p_cost, penalty2_cost]) - last_min
            atmp = cost_volume[py, px, d] + tmp_cost - last_min
            

            # update current min
            if atmp < cur_min:
                cur_min = atmp

            accum_volume[py, px, d] = atmp
            
            d += 1

        # update state variables for the next iteration
        #last_min = np.min(cur_slice)
        last_min = cur_min
        #print ptmp, direction, last_min
        #last_slice = cur_slice
        lpx = px
        lpy = py
        last_intensity = cur_intensity
        #ptmp += direction
        pyd += dy
        pxd += dx
        
        py = <unsigned int>lm.round(pyd)
        px = <unsigned int>lm.round(pxd)

    #return accum_volume
