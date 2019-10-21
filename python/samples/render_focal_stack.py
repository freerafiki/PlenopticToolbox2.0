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
Creates a focal stack, starting from a plenoptic image and its disparity map. 
--------------
v1 October 2019
@ Luca Palmieri
"""
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Read an image and create a focal stack")
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
    parser.add_argument('-spl', dest='sample_per_lens', default='15')
    parser.add_argument('-patch_shape', dest='patch_shape', default='0')
    parser.add_argument('--no_overlap', default=False, action='store_true')
    parser.add_argument('--borders', default=True, action='store_false')
    parser.add_argument('--no_conf', default=False, action='store_true')
    parser.add_argument('--view3D', default=False, action='store_true')
    parser.add_argument('--no_disp', default=False, action='store_true')
    parser.add_argument('-fpmin', dest='fp_min', default='0')
    parser.add_argument('-fpmax', dest='fp_max', default='1')
    parser.add_argument('-fstep', dest='fp_step', default='0.1')

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

    sample_per_lens = int(args.sample_per_lens)

    #views_directory =  + '/' + pic_name + '_Rendering' + '_spl' + str(sample_per_lens) + '/'
    #if not os.path.exists(views_directory):
    #    os.makedirs(views_directory)
    fs_directory = args.output_path + '/FocalStack' + '_spl' + str(sample_per_lens) + '/'
    if not os.path.exists(fs_directory):
        os.makedirs(fs_directory)
    # if args.no_disp == False:
    #     disp_directory = views_directory + 'Disps/'
    #     if not os.path.exists(disp_directory):
    #         os.makedirs(disp_directory)

    fp_min = float(args.fp_min)
    raw_images, interp_images, calibs = xmlio.load_raw_and_interp(args.colorimage_path, args.disp_path, args.config_path)
    if fp_min < 0.1:
        print("\n################")
        print("WARNING:\nIn this version of the code, it makes no sense to have negative (or zero) as the minimum focal plane, it would not create any meaningful image.")
        print("As of now, this value relates to the patch size that will be extracted from each lens in the rendering process, so a positive number is required")
        print("We set automatically the minimum focal plane at0.1, otherwise re-run and set -fpmin to the value you want. Thanks")
        fp_min = 0.1
        print("################\n")
    fp_max = float(args.fp_max)
    if fp_max > 1:
        print("\n################")
        print("WARNING:\nIn this version of the code, the maximum focal plane should less than one, otherwise it creates artifacts (it extracts black pixels between lenses).")
        print("As of now, this value relates to the patch size that will be extracted from each lens in the rendering process, so a positive number is required")
        print("We set automatically the maximum focal plane at 0.95, otherwise re-run and set -fpmax to the value you want. Thanks")
        fp_max = 1
        print("################\n")
    fp_step = float(args.fp_step)
    fps = np.arange(fp_min, fp_max, fp_step)
    print("\n***********************")
    print("* Focal Stack:        *")
    print("*                     *")
    print("* from: {:2.3f}         *".format(fp_min))
    print("* to: {:2.3f}           *".format(fp_max))
    print("* each: {:2.3f}         *".format(fp_step))
    print("*                     *")
    print("***********************\n")  
    counter = 0
    for fp in fps:
        print("{}/{}: generating view focused at {}..".format(counter+1, len(fps), fp))
        view, coarse_disp = rtxrnd.render_interp_img_at_focal_plane(raw_images, interp_images, calibs, fp, sample_per_lens, args.borders)
        # put some zeros in front in the counter number so images are ordered in the folder
        name = "{}focal_stack_{:05d}_f{:3.3f}.png".format(fs_directory, counter, fp)  
        counter += 1        
        plt.imsave(name, view)
    # if args.no_disp == False:
    #     nameD = "{}coarse_disp_{:.0f}_{:.0f}.png".format(disp_directory, x_sh, y_sh)          
    #     plt.imsave(nameD, coarse_disp)    
                
    print("Finished!")
    print("\n******************\n")

