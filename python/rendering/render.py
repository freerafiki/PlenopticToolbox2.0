
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
The idea is that if we can find the right parameters, the patch size should be 
consistent across images.
We know that if we have the diameter of the lens, the disparity can reach 
up to almost half of it, and minimum will be close to zero.
We also know that if for example disparity is close to zero, the patch size have to be really small 
If disparity would be zero (focal plane case) then 1 pixel would be enough.
If disparity would be half of the lens diameter, the patch size should be close to half of the half of the diameters
so something like half o the disparity.
We just select some values in the middle also, dividing disparity in slices.

Also note that the patch size has to be odd (because of having one central pixel)
Later we can use a radius and select circular patches and then we have more levels
"""
def get_patch_size_absolute(disp_img, lens_diameter, isReal=True):
    
    min_ps = 1
    max_ps = np.floor(lens_diameter / 2)
    if max_ps % 2 == 0:
        max_ps += 1
    number_of_different_sizes = (max_ps - min_ps) / 2 + 1
    disparray = np.asarray(disp_img)
    mean_d = np.mean(disparray) * max_ps
    ps = np.ceil(mean_d / 2).astype(int)
    
    if ps < 1:
        ps = 1

    #print("disp {0} and patch size {1}".format(mean_d, ps))

    return ps

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


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

"""
It creates a traditional image extracting patch from the lenslet image
Resolution is set to 1/4, still need to be updated to be changeable
Patch size is chosen automatically from disparity image
Using x_shift and y_shift is possible to obtain perspective shifts, i.e. different viewpoints
--------------
February 2019

"""
def generate_a_perspective_view(lenses, col_data, disp_data, min_disp, max_disp, x_shift=0, y_shift=0, isReal=True, imgname=None):
   
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
    max_ps = np.floor(central_lens.diameter / 2)
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
        
        #pdb.set_trace()
        lens = lenses[key]
        current_img = np.asarray(col_data[key])
        current_disp = np.asarray(disp_data[key])
        ps = get_patch_size_absolute(current_disp, lens.diameter, isReal)
        cen_y, cen_x = int(round(lens.pcoord[0])), int(round(lens.pcoord[1]))
        ptc_y, ptc_x = int(cen_y / factor), int(cen_x / factor)
        if min(ptc_y, ptc_x) > max_ps and ptc_y < (ref_img.shape[0]-max_ps) and ptc_x < (ref_img.shape[1]-max_ps):       
            color_img = current_img[cen-ps+y_shift:cen+ps+1+y_shift, cen-ps+x_shift:cen+ps+1+x_shift] # patch size!
            disp_simg = current_disp[cen-ps+y_shift:cen+ps+1+y_shift, cen-ps+x_shift:cen+ps+1+x_shift]
            #pdb.set_trace()
            #print("size of color_img {0}".format(color_img.shape))
            #test_img = current_img[cen-ps:cen+ps+1, cen-ps:cen+ps+1]
            #print("size without shift {0}".format(test_img.shape))
            img_big = cv2.resize(color_img, dim, interpolation = cv2.INTER_LINEAR)
            disp_big = cv2.resize(disp_simg, dim, interpolation = cv2.INTER_LINEAR) 
            count[ptc_y-hw:ptc_y+hw+1, ptc_x-hw:ptc_x+hw+1] += 1
            psimg[ptc_y-hw:ptc_y+hw+1, ptc_x-hw:ptc_x+hw+1] = ps #color_img.shape[0] * color_img.shape[1]
            ref_img[ptc_y-hw:ptc_y+hw+1, ptc_x-hw:ptc_x+hw+1, 0:3] += img_big[:,:,0:3]
            disp_ref_img[ptc_y-hw:ptc_y+hw+1, ptc_x-hw:ptc_x+hw+1] += disp_big
    
    ref_img_fnl = np.ones_like(ref_img)
    disp_ref_img_fnl = np.ones_like(disp_ref_img)
    count[(count == 0)] = 1

    for j in range(0,3):
        ref_img_fnl[:,:,j] = ref_img[:,:,j] / count 
    disp_ref_img_fnl = disp_ref_img / count   

    ref_img_fnl[np.isnan(ref_img_fnl)] = 0
    disp_ref_img_fnl[np.isnan(disp_ref_img_fnl)] = 0   
        
    return ref_img_fnl, disp_ref_img_fnl, psimg
