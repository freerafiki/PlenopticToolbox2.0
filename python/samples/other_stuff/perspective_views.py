import matplotlib.pyplot as plt
import plenopticIO.imgIO as xmlio
import pdb
import rendering.render as rtxrnd
import numpy as np
import argparse
import json
import os
#from mpl_toolkits import mplot3d

"""
Creates a refocused image. For now it is all-in-focus, but if the disparity map used would be constant, it would be focused only partly.
It uses the depth map to detect the size of a patch to extract from each micro-lens and tile them together.
It does use upscaling for small patches (lenses that exhibits low disparity) to render the image at a resolution of 1/4 of the original image. This can also be changed in the rendering file (rendering/render.py)
--------------
v1 October 2018
@ Luca Palmieri
"""
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Read an image and create a refocused version")
    parser.add_argument(dest='input_filename', nargs=1, help="Name of the lens config file")
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
    parser.add_argument('--no_overlap', default=False, action='store_true')
    parser.add_argument('--borders', default=True, action='store_false')
    
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

    max_ps = int(args.max_ps)
    layers = int(args.layers)
    min_ps = max_ps - layers  
    full_name, nothing = args.config_path.split('.xml')
    full_name2, nothing = full_name.split('_config')
    separate_names = full_name2.split('/')
    pic_name = separate_names[len(separate_names)-1]

    print("\n******************\n")
    print("Loading the scene: colored image and disparity..")
    print("\ncolored image: found at {0}..".format(args.colorimage_path))
    print("disparity map: found at {0}..".format(args.disp_path))
    lenses = xmlio.load_with_disp(args.colorimage_path, args.disp_path, args.config_path)
    
    lens_imgs = dict()
    disp_imgs = dict()
    min_d = lenses[0,0].col_img.shape[0] * 2 # just a high number
    max_d = 0

    for key in lenses:
        lens_imgs[key] = lenses[key].col_img
        disp_imgs[key] = lenses[key].disp_img
        if np.min(np.asarray(disp_imgs[key])) < min_d:
            min_d = np.min(np.asarray(disp_imgs[key]))
        if np.max(np.asarray(disp_imgs[key])) > max_d:
            max_d = np.max(np.asarray(disp_imgs[key]))

    isReal = True
    if args.scene_type == 'synth':
        isReal = False
    
    number_of_horizontal_views = int(args.horizontal_views)
    number_of_vertical_views = int(args.vertical_views)
    jump_between_views = int(args.jump_between_views)

    # We create a folder to save the views
    print("\nGenerating the views and saving them..")
    views_directory = args.output_path + '/' + pic_name + '_Views_' + str(number_of_horizontal_views) + 'x' + str(number_of_vertical_views) + '/'
    if not os.path.exists(views_directory):
        os.makedirs(views_directory)

    color_directory = views_directory + 'Color/'
    if not os.path.exists(color_directory):
        os.makedirs(color_directory)
    disp_directory = views_directory + 'Disps/'
    if not os.path.exists(disp_directory):
        os.makedirs(disp_directory)
    other_directory = views_directory + 'Other/'
    if not os.path.exists(other_directory):
        os.makedirs(other_directory)
    pcl_directory = views_directory + 'Pointclouds/' 
    if not os.path.exists(pcl_directory):
        os.makedirs(pcl_directory)


    x_left = - np.floor(number_of_horizontal_views / 2).astype(int);
    x_right = np.ceil(number_of_horizontal_views / 2).astype(int);
    y_bottom = - np.floor(number_of_vertical_views / 2).astype(int);
    y_top = np.ceil(number_of_vertical_views / 2).astype(int);

    viewcounter = 0
    
    if args.no_overlap is True:
        jump_between_views = np.ceil(np.floor(lenses[0,0].diameter / 2) / 4)

    LFtxt_path = other_directory + 'LF.txt'
    LFtxt = open(LFtxt_path, "w") 
    wrote_first_line = False

    for y_sh in range(y_bottom, y_top):
        for x_sh in range(x_left, x_right):
            
            print("generating view {0}, {1}..".format(x_sh, y_sh))
            c = 0
            x_shift = int(x_sh*jump_between_views)
            y_shift = int(y_sh*jump_between_views)

            col_img, disp, psimg = rtxrnd.generate_a_perspective_view(lenses, lens_imgs, disp_imgs, min_d, max_d, x_shift, y_shift, args.borders, isReal)

            name = "{}view_{:0>2d}_{:.0f}_{:.0f}.png".format(color_directory, viewcounter, x_sh, y_sh)
            dname = "{}disp_view_{:0>2d}_{:.0f}_{:.0f}.png".format(disp_directory, viewcounter, x_sh, y_sh)
            
            plt.imsave(name, col_img)
            plt.imsave(dname, disp, cmap='jet')
            if not wrote_first_line:
                LFtxt.write("{} {} {} {} 3\n".format(int(y_top-y_bottom), int(x_right-x_left), col_img.shape[0], col_img.shape[1]))
                wrote_first_line = True
            LFtxt.write("{} {} {}\n".format(name, y_sh-y_bottom, x_sh-x_left))
            viewcounter += 1
            

    LFtxt.close()
    psimgname = "{}patchsizeimg.png".format(other_directory)
    plt.imsave(psimgname, psimg)


    print("Finished!")
    print("\n******************\n")


