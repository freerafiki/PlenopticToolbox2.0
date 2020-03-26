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

import pdb
import matplotlib.pyplot as plt
import rendering.render as rtxrnd
import numpy as np
import os, sys


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Please give the path of the color image and disparity")

    print("\n******************\n")
    print("Creating a 3D visualization..\n")
    print("Loading the scene: colored image and disparity..")
    cur_folder = os.getcwd()
    
    # look for rgb
    attempt1 = cur_folder + '/' + sys.argv[1]
    attempt2 = sys.argv[1]
    found = False
    if os.path.exists(attempt1):
        found = True
        img_path = attempt1
    elif os.path.exists(attempt2):
        img_path = attempt2
        found = True
    if found:
        print("\ncolored image: found at {0}..".format(img_path))
        img = plt.imread(img_path)
    else:
        raise OSError("Image not found!")

    #look for disp
    attempt1 = cur_folder + '/' + sys.argv[2]
    attempt2 = sys.argv[2]
    found = False
    if os.path.exists(attempt1):
        found = True
        img_path = attempt1
    elif os.path.exists(attempt2):
        img_path = attempt2
        found = True
    if found:
        print("disparity map: found at {0}..".format(img_path))
        disp = plt.imread(img_path)
    else:
        raise OSError("Disp not found!")
    
    last_slash = img_path.rfind('/')
    v3d_directory = img_path[:last_slash] + '/showin3D'
    if not os.path.exists(v3d_directory):
        os.makedirs(v3d_directory)

    if len(sys.argv) < 4:
        scaling = (img.shape[0] + img.shape[1] ) / 2
        print("scaling not given, set automatically to", scaling)
    else:
        print("scaling set to ", sys.argv[3])
        scaling = float(sys.argv[3])

    meshname = "disp_{}_in3D".format(img_path.split("/")[-1][:-4])
    ok = rtxrnd.save_3D_view(img, disp[:,:,0], scaling, v3d_directory, meshname)

    print("Finished!")
    print("\n******************\n")