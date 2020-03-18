"""
This script is intended mainly for visualization purposes.
It is neither efficient (it re-reads and re-calculates stuff) nor optimized (it uses the old rendering algorithm that generates lower quality images and some small artifacts.)
However, at the current stage, this does not address these aspects, but it is just to visualize the disparity (with or without some post-processing) in 3 dimension, saving it in .ply (or it can be easily changed to other formats) so that it can be visualized with open-source mesh visualization software, like for example meshlab or CCloud.

Moreover, please note that this script does not reproject the disparity to a pointcloud, it just stretches the values to visualize it in 3D (like the matlab mesh visualization function), so the 3D points do not have actual meaning.

The actual 3D reprojection to have a metrically (or at least relative) correct pointcloud has a couple of more steps, that are here not taken into account.

A better version of this script is under development.
--------------
v1 October 2019
@ Luca Palmieri
"""

import matplotlib.pyplot as plt
import pdb
import plenopticIO.imgIO as xmlio
import rendering.render as rtxrnd
import numpy as np
import argparse
import json
import os
import cv2

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Read an image and create an all-in-focus version")
    parser.add_argument(dest='input_filename', nargs=1, help="Name of the lens config file")
    parser.add_argument('-col', dest='colorimage_path', default=None)
    parser.add_argument('-conf', dest='conf_path', default=None)
    parser.add_argument('-disp', dest='disp_path', default=None)
    parser.add_argument('-cfg', dest='config_path', default=None)
    parser.add_argument('-o', dest='output_path', default=None)
    parser.add_argument('-scene', dest='scene_type', default='real')
    parser.add_argument('-plus', dest='save_plus', default=False)
    parser.add_argument('-name', dest='output_name', default='defaultname')
    parser.add_argument('-patch_shape', dest='patch_shape', default='0')
    parser.add_argument('--borders', default=True, action='store_false')
    parser.add_argument('-s', dest='scaling_factor', default='1000')
    parser.add_argument('-min', dest='minimum_conf', default='0.2')
    parser.add_argument('-t', dest='confidence_threshold', default='0.5')
    parser.add_argument('-spl', dest='sample_per_lens', default='15')

    args = parser.parse_args()

    ### GETTING THE PARAMETERS FROM THE FILE
    input_file = args.input_filename[0]
    if input_file is None:
         raise OSError('Sorry we need the parameters file (.json). It should have been saved from the disparity_sample.py script \
            Please give the path when running this file (ex. python3 disparity2D.py path_of_the_file.json')
    # or the parameters.json file is provided, then we can read the missing part from there
    else:
        with open(input_file) as f:
            parameters = json.load(f)
            if args.colorimage_path is None:
                args.colorimage_path = parameters['image_path']
            if args.disp_path is None:
                args.disp_path = parameters['disp_path']
            if args.conf_path is None:
                args.conf_path = parameters['conf_path']
            if args.config_path is None:
                args.config_path = parameters['config_path']
            if args.output_path is None:
                args.output_path = parameters['output_path']

    print("\n******************\n")
    print("Creating a 3D visualization..\n")
    print("Loading the scene: colored image and disparity..")
    print("\ncolored image: found at {0}..".format(args.colorimage_path))
    print("disparity map: found at {0}..".format(args.disp_path))
    print("confidence map: found at {0}..".format(args.conf_path))
    patch_shape = int(args.patch_shape)
    isReal = True
    if args.scene_type == 'synth':
        isReal = False

    raw_images, interp_images, calibs = xmlio.load_raw_and_interp(args.colorimage_path, args.disp_path, args.config_path)
    i_counter = 0
    x_sh = 0
    y_sh = 0
    x_shift = 0
    y_shift = 0
    sample_per_lens = int(args.sample_per_lens)
    scaling = int(args.scaling_factor)

    v3d_directory = args.output_path + '/3DView_spl' + args.sample_per_lens + '/'
    if not os.path.exists(v3d_directory):
        os.makedirs(v3d_directory)

    
    #print("{}/{}: generating view {}, {}..".format(i_counter+1, tot_number, x_sh, y_sh))
    view, coarse_disp, disp = rtxrnd.render_interp_img_and_disp(raw_images, interp_images, calibs, x_shift, y_shift, sample_per_lens, args.borders)
    name = "{}view_{:03d}_{:.0f}_{:.0f}.png".format(v3d_directory, i_counter, x_sh, y_sh)
    dname = "{}disp_{:03d}_{:.0f}_{:.0f}.png".format(v3d_directory, i_counter, x_sh, y_sh)
    cdname = "{}coarse_disp_{:03d}_{:.0f}_{:.0f}.png".format(v3d_directory, i_counter, x_sh, y_sh)              
    i_counter += 1
    plt.imsave(name, view)
    plt.imsave(dname, disp)
    plt.imsave(cdname, coarse_disp)

    name3d = "3Dview_{:03d}_{:.0f}_{:.0f}".format(i_counter, x_sh, y_sh)
    namec3d = "coarse_3Dview_{:03d}_{:.0f}_{:.0f}".format(i_counter, x_sh, y_sh)
    ok = rtxrnd.save_3D_view(view, coarse_disp, scaling, v3d_directory, namec3d)
    ok = rtxrnd.save_3D_view(view, disp, scaling, v3d_directory, name3d)
    meshname = "3Dmesh_{:03d}_{:.0f}_{:.0f}".format(i_counter, x_sh, y_sh)
    #ok = rtxrnd.save_3D_mesh(view, disp, scaling, v3d_directory, meshname)

    print("Finished!")
    print("\n******************\n")