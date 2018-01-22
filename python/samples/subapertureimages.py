"""
Read a Raytrix images and switch to subaperture views
Shows the two images aside and then save the new one.
----
@version v1 - December 2017
@author Luca Palmieri
"""
import sys
sys.path.append('/Users/Palma/Documents/CAU/Code/PlenopticToolbox/python')
import argparse
import os
import json
import pdb
import matplotlib.pyplot as plt
import plenopticIO.imgIO as imgio
import rendering.render as imgrend
import numpy as np
import math


def map_from_micro_images_to_subaperture_images(lenses, lens_imgs, img_shape=None):

    assert len(lenses) == len(lens_imgs), "Number of lenses do not coincide"
    assert len(lenses) > 0, "0 lenses supplied"
    
    # ensure that the center lens is at the image origin
    first_lens = lenses[0, 0]
    if img_shape is None:
        img_shape = ((first_lens.pcoord) * 2 + 1).astype(int)

    if len(first_lens.col_img.shape) == 3:
        hl, wl, c = first_lens.col_img.shape
    else:
        hl, wl = first_lens.col_img.shape
        c = 1

    # here we create the structure for the image (circle images with the mask)
    assert hl == wl
    n = (hl - 1) / 2.0
    x = np.linspace(-n, n, hl)
    XX, YY = np.meshgrid(x, x)
    ind = np.where(XX**2 + YY**2 < first_lens.inner_radius**2)
    
    if len(first_lens.col_img.shape) == 3:
        img = np.zeros((img_shape[0], img_shape[1], c))     
    else:
        img = np.zeros((img_shape))
    
    ll = len(lenses)
    #l = 0    
    for key in lenses:

        data = np.asarray(lens_imgs[key])
        lens = lenses[key]
        ty = (YY + lens.pcoord[0] + 0.5).astype(int)    
        tx = (XX + lens.pcoord[1] + 0.5).astype(int)
        
        # ensure that the subimg is located within the image bounds
        if np.any(ty < 0) or np.any(tx < 0) or np.any(ty >= img_shape[0]) or np.any(tx >= img_shape[1]):
            continue
        
        #print(lens.pcoord)
        #print(lens.pcoord[0] / lens.col_img.shape[0], lens.pcoord[1] / lens.col_img.shape[1])
        if len(data.shape) > 0: 
        
            start_ind_x = lens.pcoord[0] / lens.col_img.shape[0]
            start_ind_y = lens.pcoord[1] / lens.col_img.shape[1]
            end_ind_x = lens.pcoord[0] / lens.col_img.shape[0] + lens.col_img.shape[0]
            end_ind_y = lens.pcoord[1] / lens.col_img.shape[1] + lens.col_img.shape[1]
            
            #pdb.set_trace()
            for (i, j, c), pixel in np.ndenumerate(lens.col_img):
                
                if c == 0:
                    current_point_x = lens.pcoord[0] - (lens.diameter/2) + i
                    current_point_y = lens.pcoord[1] - (lens.diameter/2) + j

                    lens_pos_x = math.floor(current_point_x / lens.col_img.shape[0])
                    lens_pos_y = math.floor(current_point_y / lens.col_img.shape[1])
                    
                    microview_x_length = img_shape[0] / lens.col_img.shape[0]
                    microview_y_length = img_shape[1] / lens.col_img.shape[1]
                    
                    microview_x_length = int(microview_x_length)
                    microview_y_length = int(microview_y_length)

                    x_index = (i * microview_x_length + lens_pos_x)
                    y_index = (j * microview_y_length + lens_pos_y)
                    img[x_index, y_index] = lens.col_img[i,j]

    return img, [microview_x_length, microview_y_length], lens.col_img.shape 

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Read an image and estimate disparity")
    parser.add_argument(dest='input_filename', nargs=1, help="Name of the lens config file")
    parser.add_argument('-o', dest='output_path', default='./')
    
    args = parser.parse_args()

    filename, ext = args.input_filename[0].split(".")
    config_file = filename + ".xml"
    
    if os.path.exists(args.output_path) is False:
        raise OSError('Path {0} does not exist'.format(args.output_path))

    lenses = imgio.load_from_xml(args.input_filename[0], config_file)

    lens_imgs = dict()
    for key in lenses:
        lens_imgs[key] = lenses[key].col_img
    microimg = imgrend.render_lens_imgs(lenses, lens_imgs)    
    subapertimg, dimensions, shape = map_from_micro_images_to_subaperture_images(lenses, lens_imgs) 
    plt.subplot(121)
    plt.title("MicroImages")
    plt.imshow(microimg)
    plt.subplot(122)
    plt.title("Subaperture Views")
    plt.imshow(subapertimg)
    plt.show()
    
    subapertureimg = filename + "_subarpertureimages.png"
    plt.imsave(subapertureimg, subapertimg)


