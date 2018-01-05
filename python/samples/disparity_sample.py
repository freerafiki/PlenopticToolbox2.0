"""
Sample code to read an image and estimate the disparity
Parameters can be input by hand or predefinite ones will be used
----
@veresion v1 - December 2017
@author Luca Palmieri
"""
import disparity.disparity_methods as rtxmain
import argparse
import os
import json
import pdb
import matplotlib.pyplot as plt


class EvalParameters(object):

    def __init__(self):

        self.max_disp_fac = 0.4 
        self.min_disp_fac = 0.05 
        self.max_ring = 7
        self.max_cost = 10.0
        self.penalty1 = 0.01 
        self.penalty2 = 0.03 
        self.method = 'plain'
        self.use_rings = '0,1'
        self.refine = True
        self.coc_thresh = 1.2#1.5
        self.conf_sigma = 0.2
        self.max_conf = 2.0
        self.filename = None
        self.coarse = True
        self.coarse_weight = 0.01
        self.struct_var = 0.01
        self.coarse_penalty1 = 0.01
        self.coarse_penalty2 = 0.03
        self.technique = 'sad'
        self.lut_trade_off = 1



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Read an image and estimate disparity")
    parser.add_argument(dest='input_filename', nargs=1, help="Name of the lens config file")
    parser.add_argument('-o', dest='output_path', default='./')
    parser.add_argument('--coarse', default=False, action='store_true')
    parser.add_argument('-t', dest='technique', default='sad')
    parser.add_argument('-dmin', dest='min_disp', default='1')
    parser.add_argument('-dmax', dest='max_disp', default='9')
    parser.add_argument('-nd', dest='num_disp', default='16')
    
    args = parser.parse_args()

    if os.path.exists(args.output_path) is False:
        raise OSError('Path {0} does not exist'.format(args.output_path))
                          
    params = EvalParameters()
    params.filename = args.input_filename[0]
    params.coarse = args.coarse
    params.technique = args.technique
    params.method = 'real_lut'
    params.min_disp = args.min_disp
    params.max_disp = args.max_disp
    params.num_disp = args.num_disp

    I, disp, Dwta, Dgt, Dconf, Dcoarse, sgm_err, wta_err, disparities, ncomp, disp_avg, sgm_err_mask, err_img, err_img_thresh, err_mse, new_offset = rtxmain.estimate_disp(params)

    disp_name = "{0}/disp_{1}_{2}_{3}_{4}.png".format(args.output_path, params.method, disparities[0], disparities[-1], params.technique) 
    disp_name_col = "{0}/disp_col_{1}_{2}_{3}_{4}.png".format(args.output_path, params.method, disparities[0], disparities[-1], params.technique) 

    plt.subplot(121)
    plt.title("Input Image")
    plt.imshow(I)
    plt.subplot(122)
    plt.title("Disparity Image")
    plt.imshow(disp)
    plt.show()
    
    plt.imsave(disp_name, disp, vmin=disparities[0], vmax=disparities[-1], cmap='gray')
    plt.imsave(disp_name_col, disp, vmin=disparities[0], vmax=disparities[-1])


