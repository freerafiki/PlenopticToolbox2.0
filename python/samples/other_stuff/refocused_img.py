
import matplotlib.pyplot as plt
import plenopticIO.imgIO as xmlio
import pdb
import rendering.render as rtxrnd
import numpy as np
import argparse
import os

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
    parser.add_argument('-disp', dest='disp_path', default='./')
    parser.add_argument('-cfg', dest='config_path', default='./')
    parser.add_argument('-o', dest='output_path', default='./')
    parser.add_argument('-ps', dest='max_ps', default='7')
    parser.add_argument('-lvl', dest='layers', default='4')
    parser.add_argument('-scene', dest='scene_type', default='real')
    parser.add_argument('-plus', dest='save_plus', default=False)
    parser.add_argument('-name', dest='output_name', default='')
    
    args = parser.parse_args()

    if os.path.exists(args.output_path) is False:
        raise OSError('Path {0} does not exist'.format(args.output_path))
    if os.path.exists(args.disp_path) is False:
        raise OSError('Path for disparity image: {0} does not exist'.format(args.output_path))
    if os.path.exists(args.config_path) is False:
        raise OSError('Path for configuration (.xml) file: {0} does not exist'.format(args.output_path))
        
    max_ps = int(args.max_ps)
    layers = int(args.layers)
    min_ps = max_ps - layers  
    lenses = xmlio.load_with_disp(args.input_filename[0], args.disp_path, args.config_path)

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
    print("Creating the refocused image..")
    pv, pv_disp, psimg = rtxrnd.generate_a_perspective_view(lenses, lens_imgs, disp_imgs, min_d, max_d, 0, 0, isReal)
    
    # show the images
    plt.subplot(121)
    plt.title("Refocused Image")
    plt.imshow(pv)
    plt.subplot(122)
    plt.title("Corresponding Disparity")
    plt.imshow(pv_disp)
    
    # save them
    ref_img_name = "{0}{1}_ref_img.png".format(args.output_path, args.output_name)
    ref_disp_name = "{0}{1}_ref_disp_jet.png".format(args.output_path, args.output_name)
    ref_disp_gray = "{0}{1}_ref_disp.png".format(args.output_path, args.output_name)
    plt.imsave(pv, img_aif)
    plt.imsave(pv_disp, disp_aif, cmap='jet')
    plt.imsave(pv_disp, disp_aif, cmap='gray')
    if args.save_plus:
        layers_map = "{0}{1}_layers_map_img.png".format(args.output_path, args.output_name)
        plt.imsave(layers_map, psimg, cmap='jet')
        
