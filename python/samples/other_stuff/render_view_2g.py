import matplotlib.pyplot as plt
import pdb
import plenopticIO.imgIO as xmlio
import rendering.render as rtxrnd
import numpy as np
import argparse
import json
import os
import cv2

"""
Rendering of images from a plenoptic image and its disparity map.
It renders sampling the micro-images, and it can render various configurations,
changing number of horizontal and vertical views, and the disparity range between them.
Also, changing the number of samples will affect the resolution of the rendered images.
--------------
v1 September 2019
@ Luca Palmieri
"""
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Read an image and create an all-in-focus version")
    parser.add_argument(dest='input_filename', nargs=1, help="parameters json file")
    parser.add_argument('-col', dest='colorimage_path', default=None)
    parser.add_argument('-conf', dest='conf_path', default=None)
    parser.add_argument('-disp', dest='disp_path', default=None)
    parser.add_argument('-cfg', dest='config_path', default=None)
    parser.add_argument('-numd', dest='number_of_disparities', default=12)
    parser.add_argument('-o', dest='output_path', default=None)
    parser.add_argument('-ps', dest='max_ps', default='7')
    parser.add_argument('-lvl', dest='layers', default='4')
    parser.add_argument('-scene', dest='scene_type', default='real')
    parser.add_argument('-plus', dest='save_plus', default=False)
    parser.add_argument('-name', dest='output_name', default='defaultname')
    parser.add_argument('-hv', dest='horizontal_views', default='3')
    parser.add_argument('-vv', dest='vertical_views', default='3')
    parser.add_argument('-jump', dest='jump_between_views', default='1')
    parser.add_argument('-spl', dest='sample_per_lens', default='15')
    parser.add_argument('-patch_shape', dest='patch_shape', default='0')
    parser.add_argument('--no_overlap', default=False, action='store_true')
    parser.add_argument('--borders', default=True, action='store_false')
    parser.add_argument('--no_conf', default=False, action='store_true')
    parser.add_argument('--view3D', default=False, action='store_true')
    parser.add_argument('--no_disp', default=False, action='store_true')

    
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
            if (int(args.number_of_disparities) == 0):
                disparities = parameters['disparities']
                args.number_of_disparities = len(disparities)

    #if os.path.exists(args.output_path) is False:
    #    raise OSError('Path {0} does not exist'.format(args.output_path))
    #if os.path.exists(args.disp_path) is False:
    #    raise OSError('Path for disparity image: {0} does not exist'.format(args.disp_path))
    #if os.path.exists(args.config_path) is False:
    #    raise OSError('Path for configuration (.xml) file: {0} does not exist'.format(args.config_path))
    print("\n***********************")
    print("* Settings:           *")
    print("*                     *")
    print("* Views: {}x{}          *".format(args.horizontal_views, args.vertical_views))
    print("* Creating..          *")
    if args.no_conf == False:
        print("* Confidence: True    *")
    else:
        print("* Confidence: False   *")
    if args.no_disp == False:
        print("* Disparity: True     *")
    else:
        print("* Disparity: False    *")
    print("*                     *")
    print("***********************\n")    
    max_ps = int(args.max_ps)
    layers = int(args.layers)
    patch_shape = int(args.patch_shape)
    min_ps = max_ps - layers  
    full_name, nothing = args.config_path.split('.xml')
    full_name2, nothing = full_name.split('_config')
    separate_names = full_name2.split('/')
    pic_name = separate_names[len(separate_names)-1]
    mask_path = args.colorimage_path[:-4] + '_BGmask.png'
    #pdb.set_trace()
    #print("\n******************\n")
    print("Loading the scene: colored image and disparity..")
    print("\ncolored image: found at {0}..".format(args.colorimage_path))
    print("disparity map: found at {0}..".format(args.disp_path))

    number_of_horizontal_views = int(args.horizontal_views)
    number_of_vertical_views = int(args.vertical_views)
    jump_between_views = float(args.jump_between_views)
    sample_per_lens = int(args.sample_per_lens)

    views_directory = args.output_path + '/' + pic_name + '_Rendering_' + str(number_of_horizontal_views) + 'x' + str(number_of_vertical_views) + '_j' + str(jump_between_views) + '_spl' + str(sample_per_lens) + '/'
    if not os.path.exists(views_directory):
        os.makedirs(views_directory)
    color_directory = views_directory + 'Color/'
    if not os.path.exists(color_directory):
        os.makedirs(color_directory)
    if args.no_disp == False:
        disp_directory = views_directory + 'Disps/'
        if not os.path.exists(disp_directory):
            os.makedirs(disp_directory)
    x_left = - np.floor(number_of_horizontal_views / 2).astype(int);
    x_right = np.ceil(number_of_horizontal_views / 2).astype(int);
    y_bottom = - np.floor(number_of_vertical_views / 2).astype(int);
    y_top = np.ceil(number_of_vertical_views / 2).astype(int);
    views_position = np.zeros(((y_top-y_bottom),(x_right-x_left),2))

    raw_images, interp_images, calibs = xmlio.load_raw_and_interp(args.colorimage_path, args.disp_path, args.config_path)
    i_counter = 0
    tot_number = (y_top-y_bottom) * (x_right-x_left)
    for y_sh in range(y_bottom, y_top):
        for x_sh in range(x_left, x_right):
            x_shift = (x_sh*jump_between_views)
            y_shift = (y_sh*jump_between_views)
            print("{}/{}: generating view {}, {}..".format(i_counter+1, tot_number, x_sh, y_sh))
            view, coarse_disp, disp = rtxrnd.render_interp_img_and_disp(raw_images, interp_images, calibs, x_shift, y_shift, sample_per_lens, args.borders)
            pdb.set_trace()
            name = "{}view_{:03d}_{:.0f}_{:.0f}.png".format(color_directory, i_counter, x_sh, y_sh)          
            i_counter += 1
            plt.imsave(name, view)
    if args.no_disp == False:
        nameD = "{}coarse_disp_{:.0f}_{:.0f}.png".format(disp_directory, x_sh, y_sh)          
        plt.imsave(nameD, coarse_disp)    
                
    print("Finished!")
    print("\n******************\n")

