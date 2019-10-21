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
Creates a refocused image. For now it is all-in-focus, but if the disparity map used would be constant, it would be focused only partly.
It uses the depth map to detect the size of a patch to extract from each micro-lens and tile them together.
It does use upscaling for small patches (lenses that exhibits low disparity) to render the image at a resolution of 1/4 of the original image. This can also be changed in the rendering file (rendering/render.py)
--------------
v1 October 2018
@ Luca Palmieri
"""
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Read an image and create an all-in-focus version")
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
    if args.view3D == True:
        print("* 3DViews: True       *")
    else:
        print("* 3DViews: False      *")
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

    if args.no_conf == True:
        #a = xmlio.load_and_render(args.colorimage_path, args.disp_path, args.config_path)
        lenses = xmlio.load_with_disp(args.colorimage_path, args.disp_path, args.config_path)
    else:
        print("confidence map: found at {0}..".format(args.conf_path))
        #a = xmlio.load_and_render(args.colorimage_path, args.disp_path, args.config_path)
        lenses = xmlio.load_triplet(args.colorimage_path, args.disp_path, args.conf_path, args.config_path) #, mask_path)

    min_d = 0 #lenses[0,0].col_img.shape[0] * 2 # just a high number
    max_d = 0

    isReal = True
    if args.scene_type == 'synth':
        isReal = False
        
    number_of_horizontal_views = int(args.horizontal_views)
    number_of_vertical_views = int(args.vertical_views)
    jump_between_views = int(args.jump_between_views)

    # We create a folder to save the views
    print("\nGenerating the views and saving them..")
    views_directory = args.output_path + '/' + pic_name + '_FocusedViews_' + str(number_of_horizontal_views) + 'x' + str(number_of_vertical_views) + '/'
    if not os.path.exists(views_directory):
        os.makedirs(views_directory)

    color_directory = views_directory + 'Color/'
    if not os.path.exists(color_directory):
        os.makedirs(color_directory)
    if args.no_disp == False:
        disp_directory = views_directory + 'Disps/'
        if not os.path.exists(disp_directory):
            os.makedirs(disp_directory)
    other_directory = views_directory + 'Other/'
    if not os.path.exists(other_directory):
        os.makedirs(other_directory)
    if args.no_conf == False: 
        conf_directory = views_directory + 'Confid/'
        if not os.path.exists(conf_directory):
            os.makedirs(conf_directory)

    if args.view3D == True: 
        v3d_directory = views_directory + '3DViews/'
        if not os.path.exists(v3d_directory):
            os.makedirs(v3d_directory)

    x_left = - np.floor(number_of_horizontal_views / 2).astype(int);
    x_right = np.ceil(number_of_horizontal_views / 2).astype(int);
    y_bottom = - np.floor(number_of_vertical_views / 2).astype(int);
    y_top = np.ceil(number_of_vertical_views / 2).astype(int);
    
    # I need the size of the input image and the 

    #plt.ion()
    viewcounter = 0
    
    if args.no_overlap is True:
        #pdb.set_trace()
        jump_between_views = np.ceil(np.floor(lenses[0,0].diameter / 2) / 4)

    LFtxt_path = other_directory + 'LF.txt'
    LFtxt = open(LFtxt_path, "w") 
    wrote_first_line = False

    for y_sh in range(y_bottom, y_top):
        for x_sh in range(x_left, x_right):
            
            print("generating view {0}, {1}..".format(x_sh, y_sh))
            x_shift = int(x_sh*jump_between_views)
            y_shift = int(y_sh*jump_between_views)
            image_color, initial_disp, refined_disp, patch_size_img, confidence, proc_disp = rtxrnd.generate_view_focused_micro_lenses_v2(lenses, min_d, max_d, args.no_conf, x_shift, y_shift, patch_shape, args.borders, isReal)
            name = "{}view_{:0>2d}_{:.0f}_{:.0f}.png".format(color_directory, viewcounter, x_sh, y_sh)          
            plt.imsave(name, image_color)
            #pdb.set_trace()
            mask = np.ones_like(refined_disp)
            # mask = rtxrnd.createMaskBG(image_color, [0.1, 0.4, 0.25, 0.9, 0.4, 0.9])
            # kernel = np.ones((5,5),np.uint8)
            # mask = cv2.erode(mask.astype(np.uint8),kernel,iterations = 1)
            mask3c = np.dstack((mask, mask, mask))
            if args.no_disp == False:
                dname = "{}disp_view_{:0>2d}_{:.0f}_{:.0f}.png".format(disp_directory, viewcounter, x_sh, y_sh)
                pdname = "{}proc_disp_view_{:0>2d}_{:.0f}_{:.0f}.png".format(disp_directory, viewcounter, x_sh, y_sh)
                sdname = "{}sparse_disp_view_{:0>2d}_{:.0f}_{:.0f}.png".format(disp_directory, viewcounter, x_sh, y_sh)
                plt.imsave(dname, refined_disp * mask, cmap='jet')
                plt.imsave(pdname, proc_disp * mask, cmap='jet')
                plt.imsave(sdname, refined_disp * (confidence > 0.5) * mask, cmap='jet')
            if not wrote_first_line:
                LFtxt.write("{} {} {} {} 3\n".format(int(y_top-y_bottom), int(x_right-x_left), image_color.shape[0], image_color.shape[1]))
                wrote_first_line = True
            LFtxt.write("{} {} {}\n".format(name, y_sh-y_bottom, x_sh-x_left))
            if args.no_conf == False:
                conf_name =  "{}confidence_{:0>2d}_{:.0f}_{:.0f}.png".format(conf_directory, viewcounter, x_sh, y_sh)
                plt.imsave(conf_name, confidence, cmap='gray')
            if args.view3D == True:
                sparse_disparity = refined_disp * (confidence > 0.5) * mask
                disparity = refined_disp * (confidence > 0.2) * mask
                scaling = 1000;
                sparse_pcl_name =  "sparse_disp_3Dview_{:0>2d}_{:.0f}_{:.0f}".format(viewcounter, x_sh, y_sh)
                pcl_name =  "disp_3Dview_{:0>2d}_{:.0f}_{:.0f}".format(viewcounter, x_sh, y_sh)
                proc_pcl_name =  "proc_disp_3Dview_{:0>2d}_{:.0f}_{:.0f}".format(viewcounter, x_sh, y_sh)
                ok = rtxrnd.save_3D_view(image_color, sparse_disparity, scaling, v3d_directory, sparse_pcl_name)
                ok = rtxrnd.save_3D_view(image_color, disparity, scaling, v3d_directory, pcl_name)
                ok = rtxrnd.save_3D_view(image_color, proc_disp * mask, scaling, v3d_directory, proc_pcl_name)
            viewcounter += 1

    LFtxt.close()
    psimgname = "{}patchsizeimg.png".format(other_directory)
    plt.imsave(psimgname, patch_size_img)


    print("Finished!")
    print("\n******************\n")

