
import numpy as np
import pdb
import cv2
import matplotlib.pyplot as plt
from skimage import io, color
import os

def render_lens_imgs(lenses, lens_imgs, img_shape=None):

    """
    Parameters
    ----------

    lenses: dictionary, keys are integer pairs (axial hex coordinates)
            The lens dictionary

    lens_imgs: dictionary
               Dictionary with the lens data, same size as lenses

    img_shape: pair of integers
               Shape of the target image

    Returns
    -------

    img:    array like
            Two-dimensional array containing the microlens depth image
            
    """

    assert len(lenses) == len(lens_imgs), "Number of lenses do not coincide"
    assert len(lenses) > 0, "0 lenses supplied"
    
    first_lens = lenses[0, 0]

    # ensure that the center lens is at the image origin
    if img_shape is None:
        img_shape = ((first_lens.pcoord) * 2 + 1).astype(int)
    
    # check if it is a colored image or a one-channel gray/disparity image
    if len(lens_imgs[0,0].shape) == 3:
        hl, wl, c = lens_imgs[0,0].shape
    else:
        hl, wl = lens_imgs[0,0].shape
        c = 1

    # here we create the structure for the image (circle images with the mask)
    assert hl == wl
    n = (hl - 1) / 2.0
    x = np.linspace(-n, n, hl)

    XX, YY = np.meshgrid(x, x)
    ind = np.where(XX**2 + YY**2 < first_lens.inner_radius**2)
    
    # micro image, so it takes the shape of first_lens.col_img_shape
    
    if len(lens_imgs[0,0].shape) == 3:
        img = np.zeros((img_shape[0], img_shape[1], c))
    else:
        img = np.zeros((img_shape))
     
    #img = np.zeros((lens_imgs[0,0].shape))
    
    for key in lenses:
        
        data = np.asarray(lens_imgs[key])

        lens = lenses[key]
        ty = (YY + lens.pcoord[0] + 0.5).astype(int)
        tx = (XX + lens.pcoord[1] + 0.5).astype(int)
        
        # ensure that the subimg is located within the image bounds
        if np.any(ty < 0) or np.any(tx < 0) or np.any(ty >= img_shape[0]) or np.any(tx >= img_shape[1]):
            continue

        if len(data.shape) > 0:
            img[(ty[ind], tx[ind])] = data[ind]
        else:
            img[(ty[ind], tx[ind])] = data
    

    
    return img
    
def render_cropped_img(lenses, lens_imgs, x1, y1, x2, y2):

    """
    Parameters
    ----------

    lenses: dictionary, keys are integer pairs (axial hex coordinates)
            The lens dictionary

    lens_imgs: dictionary
               Dictionary with the lens data, same size as lenses

    img_shape: pair of integers
               Shape of the target image

    Returns
    -------

    img:    array like
            Two-dimensional array containing the microlens depth image
            
    """

    assert len(lenses) == len(lens_imgs), "Number of lenses do not coincide"
    assert len(lenses) > 0, "0 lenses supplied"
    
    first_lens = lenses[0, 0]
    central_img = lens_imgs[0,0]

    # ensure that the center lens is at the image origin
    if img_shape is None:
        img_shape = ((first_lens.pcoord) * 2 + 1).astype(int)

    # check if it's gray image (disparity) or colored image
    if len(central_img.shape) == 3:
        hl, wl, c = central_img.shape
    else:
        hl, wl = central_img.shape
        c = 1

    # here we create the structure for the image (circle images with the mask)
    assert hl == wl
    n = (hl - 1) / 2.0
    x = np.linspace(-n, n, hl)

    XX, YY = np.meshgrid(x, x)
    ind = np.where(XX**2 + YY**2 < first_lens.inner_radius**2)
    
    # micro image, so it takes the shape of first_lens.col_img_shape
    if len(central_img.shape) == 3:
        img = np.zeros((img_shape[0], img_shape[1], c))
    else:
        img = np.zeros((img_shape))
    
    for key in lenses:
        
        #pdb.set_trace()
        data = np.asarray(lens_imgs[key])
        #l_type = ((-key[0] % 3) +key[1]) % 3

        lens = lenses[key]
        ty = (YY + lens.pcoord[0] + 0.5).astype(int)
        tx = (XX + lens.pcoord[1] + 0.5).astype(int)
        
        # ensure that the subimg is located within the image bounds
        if np.any(ty < 0) or np.any(tx < 0) or np.any(ty >= img_shape[0]) or np.any(tx >= img_shape[1]):
            continue

        if len(data.shape) > 0:
            img[(ty[ind], tx[ind])] = data[ind]
        else:
            img[(ty[ind], tx[ind])] = data
    

    
    return img
        

def get_patch_size_fine(disp_img, min_d, max_d, max_ps, isReal=True, layers=3):
    
    disparray = np.asarray(disp_img)
    mean_d = np.mean(disparray)
    std_d = np.std(disparray)
    step = (max_d - min_d ) / layers
    if isReal:
        ps = max_ps - layers
        for i in range(layers):
            if mean_d > min_d + step * i:
                ps += 1
    else:
        ps = max_ps
        for i in range(layers):
            if mean_d > min_d + step * i:
                ps -= 1
    
    return max(ps, 0)

"""
REFOCUSING using patches of pixels from micro-images
or total focus also, depending on the use of the actual disparity
--------------
October 2018
"""
def refocused_using_patches(lenses, col_data, disp_data, min_disp, max_disp, max_ps=5, layers = 4, isReal=True, imgname=None):
   
    if disp_data is None:
        # refocusing!
        # not ready yet
        return None
    
    # we set the patch image to be one fourth of the original, if not otherwise specified
    factor = 4 # if changing this the final resolution will change
    central_lens = lenses[0,0]
    img_shape = ((central_lens.pcoord) * 2 + 1).astype(int)
    cen = round(central_lens.img.shape[0]/2.0)
    if len(col_data[0,0].shape) > 1:
        hl, wl, c = col_data[0,0].shape
    else:
        hl, wl = central_lens.img.shape
        c = 1
    n = (hl - 1) / 2.0
    x = np.linspace(-n, n, hl)
    XX, YY = np.meshgrid(x, x)
    ref_img = np.zeros((int(img_shape[0]/factor), int(img_shape[1]/factor), c)) 
    disp_ref_img = np.zeros((int(img_shape[0]/factor), int(img_shape[1]/factor))) 
    if c == 4:
        ref_img[:,:,3] = 1 # alpha channel
    count = np.zeros((int(img_shape[0]/factor), int(img_shape[1]/factor))) 
    psimg = np.zeros((int(img_shape[0]/factor), int(img_shape[1]/factor))) 
    actual_size = round(hl / factor)
    if actual_size % 2 == 0:
        actual_size += 1
    dim = (actual_size, actual_size)
    hw = int(np.floor(actual_size/2))
    for key in lenses:
    
        lens = lenses[key]
        current_img = np.asarray(col_data[key])
        current_disp = np.asarray(disp_data[key])
        ps = get_patch_size_fine(current_disp, min_disp, max_disp, max_ps, isReal, layers)
        cen_y, cen_x = int(round(lens.pcoord[0])), int(round(lens.pcoord[1]))
        ptc_y, ptc_x = int(cen_y / factor), int(cen_x / factor)
        if min(ptc_y, ptc_x) > max_ps and ptc_y < (ref_img.shape[0]-max_ps) and ptc_x < (ref_img.shape[1]-max_ps):       
            color_img = current_img[cen-ps:cen+ps+1, cen-ps:cen+ps+1] # patch size!
            disp_simg = current_disp[cen-ps:cen+ps+1, cen-ps:cen+ps+1]
            img_big = cv2.resize(color_img, dim, interpolation = cv2.INTER_LINEAR)
            disp_big = cv2.resize(disp_simg, dim, interpolation = cv2.INTER_LINEAR) 
            count[ptc_y-hw:ptc_y+hw+1, ptc_x-hw:ptc_x+hw+1] += 1
            psimg[ptc_y-hw:ptc_y+hw+1, ptc_x-hw:ptc_x+hw+1] = ps
            ref_img[ptc_y-hw:ptc_y+hw+1, ptc_x-hw:ptc_x+hw+1, 0:3] += img_big[:,:,0:3]
            disp_ref_img[ptc_y-hw:ptc_y+hw+1, ptc_x-hw:ptc_x+hw+1] += disp_big
    
    ref_img_fnl = np.ones_like(ref_img)
    disp_ref_img_fnl = np.ones_like(disp_ref_img)
    for j in range(0,3):
        ref_img_fnl[:,:,j] = ref_img[:,:,j] / count 
    disp_ref_img_fnl = disp_ref_img / count       
        
    return ref_img_fnl, disp_ref_img_fnl, psimg   


"""
Creating a disparity image like "conventional" ones.
Basically we build on the same idea used for the refocusing (patches)
Steps:
- create a "refocused" version
- remove low confidence value 
- fill the holes
- final post-processing filter
N.B.: The final resolution may change (helpful to compare with other methods)
and the idea is to leave it open and resize the patches to fit with whatever 
resolution we are looking for (up to 1/4 of image size). If resolution is 
smaller we can fill some holes just by interpolating with neighbour in the 
downscaling process
--------------
November 2018
"""
def render_single_disparity_map(lenses, conf_method, min_disp, max_disp, image_size, parameters, max_ps=5, layers = 4, isReal=True, conf_threshold=0):
     
    #pdb.set_trace()

    # create an image of the desired resolution
    color_img_patched = np.zeros((image_size[1],image_size[0], 3))
    color_img_norm = np.zeros((image_size[1],image_size[0], 3))
    disp_with_holes = np.zeros((image_size[1],image_size[0]))
    final_disp = np.zeros((image_size[1],image_size[0]))
    conf_img_patched = np.zeros((image_size[1],image_size[0]))
    conf_img_norm = np.zeros((image_size[1],image_size[0]))
    count = np.zeros((image_size[1],image_size[0]))
    psimg = np.zeros((image_size[1],image_size[0]))

    # calculate ratio between the dimensions
    central_lens = lenses[0,0]
    cen = np.floor(central_lens.img.shape[0]/2.0).astype(int)
    img_shape = ((central_lens.pcoord) * 2 + 1).astype(int)
    ratio_x = img_shape[1] / image_size[0]
    ratio_y = img_shape[0] / image_size[1]

    # and apply to the microimage size to get patch size
    microimage_size = central_lens.img.shape
    wanted_patch_size_x = np.round(microimage_size[0]/ratio_x).astype(int)
    if wanted_patch_size_x % 2 == 0:
        wanted_patch_size_x += 1
    wanted_patch_size_y = np.round(microimage_size[1]/ratio_y).astype(int)
    if wanted_patch_size_y % 2 == 0:
        wanted_patch_size_y += 1
    wanted_patch_size = (wanted_patch_size_y, wanted_patch_size_x)
    hps_x = (wanted_patch_size_x / 2).astype(int)
    hps_y = (wanted_patch_size_y / 2).astype(int)

    # threshold for the confidence
    if conf_method == 'manual':
        threshold_conf = conf_threshold
    elif conf_method == 'automatic':
        if os.path.exists(parameters['config_path']) is False:
            raise OSError('Automatic method relies on the confidence image, to be found at {0} does not exist'.format(parameters['conf_path']))
        else:
            conf_map = plt.imread(parameters['conf_path'])
            threshold_conf = select_automatic_threshold(conf_map)
    else:
        threshold_conf == 0.5
    
    #plt.ion()
    #pdb.set_trace()
    # loop through microimages
    for key in lenses:

        # current lens object
        lens = lenses[key]
        
        # get the position we are now
        cen_x, cen_y = int(round(lens.pcoord[1])), int(round(lens.pcoord[0]))
        ptc_x, ptc_y = int(cen_y / ratio_y), int(cen_x / ratio_x)

        # check that we are inside the image range
        if min(ptc_y, ptc_x) > max_ps and ptc_x < (color_img_patched.shape[0]-max_ps) and ptc_y < (color_img_patched.shape[1]-max_ps):

            #pdb.set_trace()
            # fetch the images
            current_img = lens.col_img
            current_disp = lens.disp_img
            current_conf = lens.conf_img

            # get the patch size for the current disparity
            ps = get_patch_size_fine(current_disp, min_disp, max_disp, max_ps, isReal, layers)

            # extract the patches
            color_img = current_img[cen-ps:cen+ps+1, cen-ps:cen+ps+1,0:3]
            disp_img = current_disp[cen-ps:cen+ps+1, cen-ps:cen+ps+1]
            conf_img = current_conf[cen-ps:cen+ps+1, cen-ps:cen+ps+1]

            # resize from the patch size to the wanted patch size
            color_resized = cv2.resize(color_img, wanted_patch_size, interpolation = cv2.INTER_LINEAR)
            conf_resized = cv2.resize(conf_img, wanted_patch_size, interpolation = cv2.INTER_LINEAR)

            # remove low confidence values
            high_conf_mask = (conf_img > threshold_conf)
            disp_high_conf = disp_img * high_conf_mask
            disp_resized = cv2.resize(disp_high_conf, wanted_patch_size, interpolation = cv2.INTER_LINEAR)
            disp2_resized = cv2.resize(disp_img, wanted_patch_size, interpolation = cv2.INTER_LINEAR)

            # fill the images
            count[ptc_x-hps_x:ptc_x+hps_x+1, ptc_y-hps_y:ptc_y+hps_y+1] += 1
            psimg[ptc_x-hps_x:ptc_x+hps_x+1, ptc_y-hps_y:ptc_y+hps_y+1] = ps
            color_img_patched[ptc_x-hps_x:ptc_x+hps_x+1, ptc_y-hps_y:ptc_y+hps_y+1, 0:3] += color_resized[:,:,0:3]
            disp_with_holes[ptc_x-hps_x:ptc_x+hps_x+1, ptc_y-hps_y:ptc_y+hps_y+1] += disp_resized
            final_disp[ptc_x-hps_x:ptc_x+hps_x+1, ptc_y-hps_y:ptc_y+hps_y+1] += disp2_resized
            conf_img_patched[ptc_x-hps_x:ptc_x+hps_x+1, ptc_y-hps_y:ptc_y+hps_y+1] += conf_resized

            #plt.subplot(121), plt.imshow(color_img)
            #plt.subplot(122), plt.imshow(current_img)
            #print(ps)


    #pdb.set_trace()

    for j in range(0,3):
        color_img_norm[:,:,j] = color_img_patched[:,:,j] / count 
    disp_with_holes_norm = disp_with_holes / count
    full_disp_norm = final_disp / count
    conf_img_norm = conf_img_patched / count

    # intermediate results
    plt.subplot(131), plt.imshow(color_img_norm)
    plt.subplot(132), plt.imshow(disp_with_holes_norm, cmap='jet')
    plt.subplot(133), plt.imshow(full_disp_norm, cmap='jet')
    plt.show()

    ## here we have an image
    disparities = np.asarray(parameters['disparities'])
    iterative_filled = fill_iterative(disp_with_holes_norm, color_img_norm, conf_img_norm, disparities)
    pdb.set_trace()
    filter_disp = disparity_filter(full_disp_norm)

    

    pdb.set_trace()
    #ref_img_fnl = np.ones_like(ref_img)
    #disp_ref_img_fnl = np.ones_like(disp_ref_img)
    #for j in range(0,3):
    #    ref_img_fnl[:,:,j] = ref_img[:,:,j] / count 
    #disp_ref_img_fnl = disp_ref_img / count       
        
    #color_img_patched, disp_with_holes, final_disp

    return color_img_patched, disp_with_holes, final_disp


"""
The idea is to iterative fill the disparity map, that it has holes (no values)
where the confidence was too low, using the other values (we assume they are correct or almost)
Using an iterative approach may be slower but allows for better debug (we can see at each iteration)
and for a better result
----
December 2018
"""
def fill_iterative(disparity_map, color_image, confidence_image, disparities):

    #pdb.set_trace()
    #correct for nan pixels
    disparity_map[np.isnan(disparity_map)] = 0
    color_image[np.isnan(color_image)] = 0
    confidence_image[np.isnan(confidence_image)] = 0

    # get the point that have no value
    empty_points = (disparity_map == 0)
    tot_number_of_points_to_be_filled = np.sum(empty_points)
    tot_number_of_points_filled = 0

    # control variable
    finished = False
    numOfIterations = 0
    win_size = 31
    hws = np.floor(win_size / 2).astype(int)

    # criterium
    thresh_empty_points = 0.05 #5 % 
    max_num_iterations = 100
    failed_iterations = 0
    max_failed_iterations = 3
    kernel = np.ones((5,5),np.uint8)
    percentage_min_filled_point_at_iteration = 0.75
    min_ratio_to_fill_a_pixel = 0.7
    diff_color_similarity = 0.25
    min_ratio_of_similar_colored_disp_values = 0.25
    constraints = [min_ratio_to_fill_a_pixel, diff_color_similarity, min_ratio_of_similar_colored_disp_values]
    margin_x = 10
    margin_y = 10

    # debug
    plt.ion()
    plt.subplot(221); plt.imshow(disparity_map)
    tmp_control = disparity_map.copy()

    # loop until you filled
    while finished is not True:

        print("Iteration {0}..".format(numOfIterations + 1))
        plt.subplot(223); plt.imshow(disparity_map)
        # get the points to fill at this iteration
        erosion = cv2.erode((empty_points).astype(np.float64),kernel,iterations = 1)
        empty_points_close_to_values = empty_points - erosion

        indices_points_to_fill = np.where(empty_points_close_to_values > 0)
        how_many_points = len(indices_points_to_fill[0])
        filled_points_at_this_iteration = 0
        for j in range(0, how_many_points):
            x = indices_points_to_fill[0][j]
            y = indices_points_to_fill[1][j]
            padding_x = max(margin_x, hws)
            padding_y = max(margin_y, hws)
            if x > padding_x and y > padding_y and x < disparity_map.shape[0] - padding_x and y < disparity_map.shape[1] - padding_y:
                colored_win = color_image[x-hws:x+hws+1, y-hws:y+hws+1]
                disparity_win = disparity_map[x-hws:x+hws+1, y-hws:y+hws+1]
                confidence_win = confidence_image[x-hws:x+hws+1, y-hws:y+hws+1]
                # try to fill hte value
                filledValue, value = fillValue(disparity_win, colored_win, confidence_win, disparities, constraints)
                if filledValue is True:
                    filled_points_at_this_iteration += 1
                    disparity_map[x,y] = value
                    #print("filled one point, d[{0}, {1}] = {2}".format(x,y, value))
        # if we see that is not getting filled, we need to relax constraint and accept lower reliable filling
        percentage_filled_point_this_iteration = filled_points_at_this_iteration / how_many_points
        print("Filled {0} points out of {1}, {2}%".format(filled_points_at_this_iteration, how_many_points, percentage_filled_point_this_iteration * 100))
        if percentage_filled_point_this_iteration < percentage_min_filled_point_at_iteration:
            constraints = relax_constraints(constraints)
            print("that was poor. we relax the constraints to fill more points")

        numOfIterations += 1
        empty_points = (disparity_map == 0)
        if np.sum(empty_points) / np.sum(disparity_map > 0) < thresh_empty_points:
            print("Filled the disparity!")
            finished = True
        if numOfIterations > max_num_iterations:
            print("passed max number of iterations")
            finished = True
            break
        if percentage_filled_point_this_iteration < 0.001:
            failed_iterations += 1
        if failed_iterations > max_failed_iterations:
            print("not doing anything anymore.. did not work")
            finished = True
            break
        plt.subplot(224); plt.imshow(disparity_map)
        plt.subplot(222); plt.imshow(disparity_map - tmp_control)
        tot_number_of_points_filled += filled_points_at_this_iteration
        print("Cumulatevely, filled {0} points out of {1}, so we are at {2}% of the work".format(tot_number_of_points_filled, tot_number_of_points_to_be_filled, (tot_number_of_points_filled / tot_number_of_points_to_be_filled ) * 100))
        #pdb.set_trace()

    return disparity_map

"""
Fill the central value (that should be 0)
using the adjacent values
If there are not enough disparity value with similar colors, 
value is not filled
Third parameters constraints is used to understand if t
the value should be filled or not

Filling process:
indicative value = mean value of neighbouring similarly-colored pixels
with a range of +- 2 from indicative value, compute energy term
E = E(color) + E(disp) = diff(color) + diff(disp) 
using the available values between the 8 neighbours (3x3 window)
and check minimum and assign that value (the value of disparity that correspond to min energy)

it returns

bool    - True or False         - if worked or not
float   - value                 - the value with which it should be filled
"""

def fillValue(disp_window, color_window, confidence_window, disparities, constraints):

    #pdb.set_trace()
    ## Check if it's allright
    win_size = disp_window.shape[0]
    assert color_window.shape[0] == win_size, "different window size!" 
    cen_pix = np.floor(win_size / 2).astype(int)
    #if disp_window[cen_pix, cen_pix] > 0:
    #    print("already filled!")
    #    return 1, disp_window[cen_pix, cen_pix]

    normalization_factor_window = np.power(win_size,2)
    mask_disp_values = (disp_window > 0)
    # check if enough disparity values to be filled
    #if np.sum(mask_disp_values) / normalization_factor_window < constraints[0]:
    #    return 0, 0

    #pdb.set_trace()
    # TODO: use three channel
    color_window_L = rgb2gray(color_window)
    color_of_the_center = color_window_L[cen_pix, cen_pix] 
    limit_different_color = constraints[1]
    #if np.max(color_window_L) > 1:
    #    limit_different_color = limit_different_color * 255
    window_with_center_value = np.ones((color_window.shape[0], color_window.shape[1])) * color_of_the_center
    mask_colored_values = ((window_with_center_value - color_window_L) > limit_different_color)
    # check if enough similarly-colored values to be filled
    #if np.sum(mask_colored_values) / normalization_factor_window < constraints[2]:
    #    return 0, 0

    # fill using idea explained above
   # disparity_values_of_similarly_colored_pixels = disp_window * mask_colored_values
   # normalization_value = np.sum(disparity_values_of_similarly_colored_pixels > 0)
   # indicative_value = np.sum(disparity_values_of_similarly_colored_pixels) / normalization_value

    sigma1 = 50
    sigma2 = 0.5
    valid_disp_values = disp_window * (disp_window > 0) 
    weights1 = np.exp(-np.abs(color_window_L - color_of_the_center) / sigma1)
    weights2 = np.exp(-(1/confidence_window) / sigma2)
    weighted_valid = valid_disp_values * weights1 * weights2
    norm_fact = np.sum((weighted_valid > 0))
    indicative_value = np.sum(valid_disp_values) / norm_fact
    #pdb.set_trace()
    """
    plt.subplot(221)
    plt.imshow(disp_window)
    plt.subplot(222)
    plt.imshow(valid_disp_values)
    plt.subplot(223)
    plt.imshow(color_window)
    plt.subplot(224)
    plt.imshow(np.ones_like(disp_window) * indicative_value, vmin = np.min(disp_window), vmax = np.max(disp_window))
    pdb.set_trace()
    """

    # find the closest disparities to the indicative value, evaluate energy and pick best
    # TODO
    #print("this time we got a value: {0}".format(indicative_value))
    return True, indicative_value

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

"""
Constraints contains some conditions for a window around a pixel to be considered good enough for the filling
In the case in which there are not such condition, we can relax this constraints in order to allow a dense
filling (or leave the disparity with holes)

The constraints are:
- min_ratio_to_fill_a_pixel = minimum number of pixels that have disparity value within a window 
                              (calculated in percentage over the number of pixel in the window)
- diff_color_similarity     = difference between pixels to be considerated similarly colored
                              (calculated over the range, so 0.5 usint int [0,255] would be 128)
- min_ratio_of_similar_colored_disp_values = minimum number of pixels with similar color within the window
                              (calculated in percentage over the number of pixel in the window) 
"""
def relax_constraints(constraints):

    # TODO think about a better way, come on
    constraints[0] = 0.51 # half of the window with values: is this asking too much?
    constraints[1] = 0.5 # half of the color range, it's a lot!
    constraints[2] = 0.01 # one percent of the window (so typically one pixel is enough)
    return constraints

"""
The idea is to check how many points will be accepted with the mean, and adjust the threshold
to accept more or less 75% of the points
"""
def select_automatic_threshold(confidence_map):

    cut_value = np.mean(confidence_map) + np.std(confidence_map)

    total = np.sum(confidence_map > 0)
    accepted = np.sum(confidence_map > cut_value)
    ratio = accepted / total
    step = (np.max(confidence_map) - np.min(confidence_map)) / 10
    num_iteration = 0
    converged = False
    while ratio > 0.80 or ratio < 0.70 and not converged:
        if ratio > 0.80:
            cut_value = cut_value + step
        elif ratio < 0.70:
            cut_value = cut_value - step
        num_iteration += 1
        accepted = np.sum(confidence_map > cut_value)
        ratio = accepted / total
        if ratio > 0.70 and ratio < 0.80:
            converged = True
        if num_iteration > 5 and not converged:
            step = step / 5
        # manually break if it didn't work
        if num_iteration > 15:
            print("automatic method for thresholding did not work: selecting manually threhsold = 0.5")
            converged = True
            cut_value = 0.5
            break

    return cut_value