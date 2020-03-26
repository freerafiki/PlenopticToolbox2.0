#!/usr/local/bin/python3
import sys, os
import pdb
import matplotlib.pyplot as plt
import numpy as np
import plenopticIO.imgIO as pio
import plenopticIO.lens_grid as rtxhexgrid
import microlens.lens as rtxlens
import argparse, json

def _hex_focal_type(c):
    
    """
    Calculates the focal type for the three lens hexagonal grid
    """

    focal_type = ((-c[0] % 3) + c[1]) % 3

    return focal_type 

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Read an image and create an all-in-focus version")
    parser.add_argument(dest='input_filename', nargs=1, help="parameters json file")
    parser.add_argument('-col', dest='colorimage_path', default=None)
    parser.add_argument('-conf', dest='conf_path', default=None)
    parser.add_argument('-disp', dest='disp_path', default=None)
    parser.add_argument('-cfg', dest='config_path', default=None)
    parser.add_argument('-o', dest='output_path', default=None)
    parser.add_argument('--nomask', dest="nomask", default=False, action='store_true')
    parser.add_argument('--lt', dest="lens_types_mask", default=False, action='store_true')
    parser.add_argument('--rgb', dest="rgbmasked", default=False, action='store_true')
    parser.add_argument('--disp', dest="dispmasked", default=False, action='store_true')
    parser.add_argument('--conf', dest="confmasked", default=False, action='store_true')
    args = parser.parse_args()

    #pdb.set_trace()
    ### GETTING THE PARAMETERS FROM THE FILE
    input_file = args.input_filename[0]
    if input_file is None:
         raise OSError('Sorry we need the parameters file (.json). It should have been saved from the disparity_sample.py script \
            Please give the path when running this file (ex. python3 disparity2D.py path_of_the_file.json')
    # or the parameters.json file is provided, then we can read the missing part from there
    else:
        info = dict()
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
    if not ( (not args.nomask)
        or args.lens_types_mask
        or args.rgbmasked
        or args.dispmasked
        or args.confmasked):
        print("Nothing to do here! You did not select any mask! Have a nice day!")
    else:
        print("Reading images..")
        if args.rgbmasked:
            img = plt.imread(args.colorimage_path)
        if args.dispmasked:
            disp = plt.imread(args.disp_path)
        if args.confmasked:
            conf = plt.imread(args.conf_path)
        calib = pio.read_calibration(args.config_path)
        img_shape = np.asarray(img.shape[0:2])
        if not args.nomask:
            mi_mask_img = np.zeros(img_shape)
        if args.lens_types_mask:
            mi_mask_lt = np.zeros((img_shape[0], img_shape[1], 3))
        if args.rgbmasked:
            mi_masked_rgb = np.zeros((img_shape[0], img_shape[1], 3))
        if args.dispmasked:
            mi_masked_disp = np.zeros((img_shape[0], img_shape[1], 3))
        if args.confmasked:
            mi_masked_conf = np.zeros((img_shape[0], img_shape[1], 3))
        
        print("setting up everything..")
        coords = rtxhexgrid.hex_lens_grid(img_shape, calib.lens_diameter, calib.rot_angle, calib.offset, calib.lbasis)
        local_grid = rtxlens.LocalLensGrid(calib.lens_diameter)
        x, y = local_grid.x, local_grid.y
        xx, yy = local_grid.xx, local_grid.yy
        mask = np.zeros_like(local_grid.xx)
        mask[xx**2 + yy**2 < calib.inner_lens_radius**2] = 1  
        mask3c = np.dstack((mask, mask, mask))
        inside = np.zeros_like(local_grid.xx)
        inside[xx**2 + yy**2 < (calib.inner_lens_radius + 2)**2] = 1
        outside = np.zeros_like(local_grid.xx)
        outside[xx**2 + yy**2 > (calib.inner_lens_radius - 1)**2] = 1
        boundaries = inside * outside

        print("creating the masks..")
        for lc in coords:

            ft = _hex_focal_type(lc)
            pc = coords[lc]
            cen_x = round(pc[0])
            cen_y = round(pc[1])
            x1 = int(cen_x + round(np.min(x)))
            x2 = int(cen_x + round(np.max(x)))
            y1 = int(cen_y + round(np.min(y)))
            y2 = int(cen_y + round(np.max(y)))
            if not args.nomask:
                mi_mask_img[x1:x2+1, y1:y2+1] += mask
            if args.lens_types_mask:
                mi_mask_lt[x1:x2+1, y1:y2+1, ft] += mask 
            if args.rgbmasked:
                mi_masked_rgb[x1:x2+1, y1:y2+1,:3] += img[x1:x2+1, y1:y2+1,:3] * mask3c
                mi_masked_rgb[x1:x2+1, y1:y2+1, ft] *= (1-boundaries)
                mi_masked_rgb[x1:x2+1, y1:y2+1, ft] += (boundaries)
            if args.dispmasked:
                mi_masked_disp[x1:x2+1, y1:y2+1,:3] += disp[x1:x2+1, y1:y2+1,:3] * mask3c
                mi_masked_disp[x1:x2+1, y1:y2+1, ft] *= (1-boundaries)
                mi_masked_disp[x1:x2+1, y1:y2+1, ft] += (boundaries)
            if args.confmasked:
                mi_masked_conf[x1:x2+1, y1:y2+1,:3] += conf[x1:x2+1, y1:y2+1,:3] * mask3c
                mi_masked_conf[x1:x2+1, y1:y2+1, ft] *= (1-boundaries)
                mi_masked_conf[x1:x2+1, y1:y2+1, ft] += (boundaries)

        print("Saving..")
        output_directory = args.output_path
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        masks_directory = output_directory + '/Masks/'
        if not os.path.exists(masks_directory):
            os.makedirs(masks_directory)
        if not args.nomask:
            plt.imsave(masks_directory + "mask.png", np.clip(mi_mask_img,0,1), cmap='gray')
        if args.lens_types_mask:
            plt.imsave(masks_directory + "lens_types_mask.png", mi_mask_lt)
        if args.rgbmasked:
            plt.imsave(masks_directory + "rgb_masked.png", mi_masked_rgb)
        if args.dispmasked:
            plt.imsave(masks_directory + "disp_masked.png", mi_masked_disp)
        if args.confmasked:
            plt.imsave(masks_directory + "conf_masked.png", mi_masked_conf)
        