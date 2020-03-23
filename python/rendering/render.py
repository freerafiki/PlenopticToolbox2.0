
import numpy as np
import pdb
#import cv2
import matplotlib.pyplot as plt
import rendering.filters as filters
from skimage import io, color
import os
#from cv2.ximgproc import *
import scipy.interpolate as sinterp

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
    ps = np.ceil(mean_d * 0.5).astype(int)
    
    if ps < 1:
        ps = 1

    #print("disp {0} and patch size {1}".format(mean_d, ps))

    return ps

def get_patch_size_absolute_focused_lenses(disp_img, lens_diameter, isReal=True):
    
    min_ps = 5
    max_ps = np.floor(lens_diameter / 2)
    if max_ps % 2 == 0:
        max_ps += 1
    number_of_different_sizes = (max_ps - min_ps) / 2 + 1
    disparray = np.asarray(disp_img)
    mean_d = np.mean(disparray) * max_ps
    ps = np.round(max_ps * 0.4).astype(int)
    #ps = np.round(max_ps - (mean_d)).astype(int)
    
    if ps < 1:
        ps = 1

    #print("disp {0} and patch size {1}".format(mean_d, ps))
    #ps = np.round(ps * 1.75).astype(int)
    #if ps > max_ps:
    #    ps = max_ps
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
def generate_a_perspective_view(lenses, col_data, disp_data, min_disp, max_disp, x_shift=0, y_shift=0, cutBorders=True, isReal=True, imgname=None):
   
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
    
    if cutBorders is True:

        paddingToAvoidBorders = int(max_ps + 1)
        ref_img_fnl = ref_img_fnl[paddingToAvoidBorders:ref_img_fnl.shape[0]-paddingToAvoidBorders, paddingToAvoidBorders:ref_img_fnl.shape[1]-paddingToAvoidBorders, :]
        disp_ref_img_fnl = disp_ref_img_fnl[paddingToAvoidBorders:disp_ref_img_fnl.shape[0]-paddingToAvoidBorders, paddingToAvoidBorders:disp_ref_img_fnl.shape[1]-paddingToAvoidBorders]
        psimg = psimg[paddingToAvoidBorders:psimg.shape[0]-paddingToAvoidBorders, paddingToAvoidBorders:psimg.shape[1]-paddingToAvoidBorders]

    return ref_img_fnl, disp_ref_img_fnl, psimg


### It just generates three images (usually colored image, disparity and confidence 
### but can be used with whatever is loaded with load_triplet)
def generate_a_perspective_view_triplet(lenses, x_shift=0, y_shift=0, cutBorders=True, isReal=True, imgname=None):

    # we set the patch image to be one fourth of the original, if not otherwise specified
    factor = 4 # if changing this the final resolution will change
    central_lens = lenses[0,0]
    img_shape = ((central_lens.pcoord) * 2 + 1).astype(int)
    cen = round(central_lens.img.shape[0]/2.0)
    if len(lenses[0,0].col_img.shape) > 1:
        hl, wl, c = lenses[0,0].col_img.shape
    else:
        hl, wl = central_lens.img.shape
        c = 1
    max_ps = np.floor(central_lens.diameter / 2)
    n = (hl - 1) / 2.0
    x = np.linspace(-n, n, hl)
    XX, YY = np.meshgrid(x, x)
    ref_img = np.zeros((int(img_shape[0]/factor), int(img_shape[1]/factor), c)) 
    # we assume they don't have colors! (usually disp and confidence do not have channels..)
    ref_disp = np.zeros((int(img_shape[0]/factor), int(img_shape[1]/factor))) 
    ref_conf = np.zeros((int(img_shape[0]/factor), int(img_shape[1]/factor))) 
    if c == 4:
        ref_img[:,:,3] = 1 # alpha channel
    count = np.zeros((int(img_shape[0]/factor), int(img_shape[1]/factor))) 
    actual_size = round(hl / factor)
    if actual_size % 2 == 0:
        actual_size += 1
    dim = (actual_size, actual_size)
    hw = int(np.floor(actual_size/2))
    for key in lenses:
        
        #pdb.set_trace()
        lens = lenses[key]
        current_img = np.asarray(lens.col_img)
        current_disp = np.asarray(lens.disp_img)
        current_conf = np.asarray(lens.conf_img)
        ps = get_patch_size_absolute(current_disp, lens.diameter, isReal)
        cen_y, cen_x = int(round(lens.pcoord[0])), int(round(lens.pcoord[1]))
        ptc_y, ptc_x = int(cen_y / factor), int(cen_x / factor)
        if min(ptc_y, ptc_x) > max_ps and ptc_y < (ref_img.shape[0]-max_ps) and ptc_x < (ref_img.shape[1]-max_ps):       
            color_img = current_img[cen-ps+y_shift:cen+ps+1+y_shift, cen-ps+x_shift:cen+ps+1+x_shift] # patch size!
            disp_simg = current_disp[cen-ps+y_shift:cen+ps+1+y_shift, cen-ps+x_shift:cen+ps+1+x_shift]
            conf_simg = current_conf[cen-ps+y_shift:cen+ps+1+y_shift, cen-ps+x_shift:cen+ps+1+x_shift]#pdb.set_trace()
            #print("size of color_img {0}".format(color_img.shape))
            #test_img = current_img[cen-ps:cen+ps+1, cen-ps:cen+ps+1]
            #print("size without shift {0}".format(test_img.shape))
            img_big = cv2.resize(color_img, dim, interpolation = cv2.INTER_LINEAR)
            disp_big = cv2.resize(disp_simg, dim, interpolation = cv2.INTER_LINEAR) 
            conf_big = cv2.resize(conf_simg, dim, interpolation = cv2.INTER_LINEAR) 
            count[ptc_y-hw:ptc_y+hw+1, ptc_x-hw:ptc_x+hw+1] += 1
            ref_img[ptc_y-hw:ptc_y+hw+1, ptc_x-hw:ptc_x+hw+1, 0:3] += img_big[:,:,0:3]
            ref_disp[ptc_y-hw:ptc_y+hw+1, ptc_x-hw:ptc_x+hw+1] += disp_big
            ref_conf[ptc_y-hw:ptc_y+hw+1, ptc_x-hw:ptc_x+hw+1] += conf_big
    
    ref_img_fnl = np.ones_like(ref_img)
    #disp_ref_img_fnl = np.ones_like(disp_ref_img)
    count[(count == 0)] = 1

    for j in range(0,3):
        ref_img_fnl[:,:,j] = ref_img[:,:,j] / count 
    disp_ref_img_fnl = ref_disp / count   
    conf_ref_img_fnl = ref_conf / count  

    ref_img_fnl[np.isnan(ref_img_fnl)] = 0
    disp_ref_img_fnl[np.isnan(disp_ref_img_fnl)] = 0  
    conf_ref_img_fnl[np.isnan(conf_ref_img_fnl)] = 0    
    
    if cutBorders is True:

        paddingToAvoidBorders = int(max_ps + 1)
        ref_img_fnl = ref_img_fnl[paddingToAvoidBorders:ref_img_fnl.shape[0]-paddingToAvoidBorders, paddingToAvoidBorders:ref_img_fnl.shape[1]-paddingToAvoidBorders, :]
        disp_ref_img_fnl = disp_ref_img_fnl[paddingToAvoidBorders:disp_ref_img_fnl.shape[0]-paddingToAvoidBorders, paddingToAvoidBorders:disp_ref_img_fnl.shape[1]-paddingToAvoidBorders]
        conf_ref_img_fnl = conf_ref_img_fnl[paddingToAvoidBorders:conf_ref_img_fnl.shape[0]-paddingToAvoidBorders, paddingToAvoidBorders:conf_ref_img_fnl.shape[1]-paddingToAvoidBorders]
        #psimg = psimg[paddingToAvoidBorders:psimg.shape[0]-paddingToAvoidBorders, paddingToAvoidBorders:psimg.shape[1]-paddingToAvoidBorders]

    return ref_img_fnl, disp_ref_img_fnl, conf_ref_img_fnl

"""
Createas a view using only micro-lenses that are on focus
doing so, spatial resolution is reduced but also blur and artifacts.

It first creates three images using only one lens type, then pick the part of 
those images that are in focus and merge them together using a weighted average

the idea is that by averaging them together you reduce artifacts (in shiny parts and edges)
but by using weights (so weight more the ones that are in focus) you keep the sharpness

--------------
February 2019

"""

def generate_view_focused_micro_lenses(lenses, min_disp=0, max_disp=0, no_conf=False, x_shift=0, y_shift=0, patch_shape=0, cutBorders=True, isReal=True, imgname=None):
   
    # bilateral filter

    triplet = [[12, 5, 7], [10, 7, 9], [8, 11, 13], [6, 13, 15], [4, 15, 17]]
    chosen = 3
    # we set the patch image to be one/eigth of the original, if not otherwise specified
    factor = triplet[chosen][0] # if changing this the final resolution will change
    central_lens = lenses[0,0]
    if max_disp == 0:
        max_disp = central_lens.diameter
    img_shape = ((central_lens.pcoord) * 2 + 1).astype(int)
    cen = round(central_lens.img.shape[0]/2.0)
    if len(central_lens.col_img.shape) > 1:
        hl, wl, c = central_lens.col_img.shape
    else:
        hl, wl = central_lens.img.shape
        c = 1
    max_ps = np.floor(central_lens.diameter / 2)
    img_lens_type0 = np.zeros((int(img_shape[0]/factor), int(img_shape[1]/factor), c)) 
    img_lens_type1 = np.zeros((int(img_shape[0]/factor), int(img_shape[1]/factor), c)) 
    img_lens_type2 = np.zeros((int(img_shape[0]/factor), int(img_shape[1]/factor), c)) 
    if c == 4:
        img_lens_type0[:,:,3] = 1 # alpha channel
        img_lens_type1[:,:,3] = 1 # alpha channel
        img_lens_type2[:,:,3] = 1 # alpha channel
    disp_lens_type0 = np.zeros((int(img_shape[0]/factor), int(img_shape[1]/factor))) 
    disp_lens_type1 = np.zeros((int(img_shape[0]/factor), int(img_shape[1]/factor))) 
    disp_lens_type2 = np.zeros((int(img_shape[0]/factor), int(img_shape[1]/factor))) 
    if no_conf == False:
        conf_lens_type0 = np.zeros((int(img_shape[0]/factor), int(img_shape[1]/factor))) 
        conf_lens_type1 = np.zeros((int(img_shape[0]/factor), int(img_shape[1]/factor))) 
        conf_lens_type2 = np.zeros((int(img_shape[0]/factor), int(img_shape[1]/factor))) 
    count0 = np.zeros((int(img_shape[0]/factor), int(img_shape[1]/factor))) 
    count1 = np.zeros((int(img_shape[0]/factor), int(img_shape[1]/factor))) 
    count2 = np.zeros((int(img_shape[0]/factor), int(img_shape[1]/factor))) 
    psimg0 = np.zeros((int(img_shape[0]/factor), int(img_shape[1]/factor))) 
    psimg1 = np.zeros((int(img_shape[0]/factor), int(img_shape[1]/factor))) 
    psimg2 = np.zeros((int(img_shape[0]/factor), int(img_shape[1]/factor))) 
    actual_size_x = triplet[chosen][1] #15
    actual_size_y = triplet[chosen][2] #round(hl / factor) + 4
    if actual_size_x % 2 == 0:
        actual_size_x += 1
    dim = (actual_size_x, actual_size_y)
    hw_x = int(np.floor(actual_size_x/2))
    hw_y = int(np.floor(actual_size_y/2))
    # create a mask to actual extract eclipses patches
    radius = np.floor(actual_size_y/2)
    x = np.linspace(-1, 1, actual_size_y) * radius
    xx, yy = np.meshgrid(x, x)
    if patch_shape == 0:
        rect_mask = np.ones_like(xx)
        mask = rect_mask[:,1:rect_mask.shape[1]-1]
    elif patch_shape == 1:
        circle_mask = np.zeros_like(xx)
        circle_mask[xx**2 + yy**2 < (radius+1)**2] = 1
        mask = circle_mask[:,1:circle_mask.shape[1]-1]
    mask4c = np.dstack((mask, mask, mask, mask))

    # loop and create three images!
    for key in lenses:
        
        #pdb.set_trace()
        lens = lenses[key]
        current_img = np.asarray(col_data[key])
        current_disp = np.asarray(disp_data[key])
        if no_conf == False:
            current_conf = np.asarray(conf_data[key])
        ps = get_patch_size_absolute_focused_lenses(current_disp, lens.diameter, isReal)
        cen_y, cen_x = int(np.round(lens.pcoord[0])), int(np.floor(lens.pcoord[1]))
        ptc_y, ptc_x = int(cen_y / factor), int(cen_x / factor)
        if min(ptc_y, ptc_x) > max_ps and ptc_y < (img_lens_type0.shape[0]-max_ps) and ptc_x < (img_lens_type0.shape[1]-max_ps):       
            color_img = current_img[cen-ps+y_shift:cen+ps+1+y_shift, cen-ps+x_shift:cen+ps+1+x_shift] # patch size!
            disp_simg = current_disp[cen-ps+y_shift:cen+ps+1+y_shift, cen-ps+x_shift:cen+ps+1+x_shift]
            if no_conf == False:
                conf_img = current_conf[cen-ps+y_shift:cen+ps+1+y_shift, cen-ps+x_shift:cen+ps+1+x_shift]
            img_big = cv2.resize(color_img, dim, interpolation = cv2.INTER_LINEAR) * mask4c
            disp_big = cv2.resize(disp_simg, dim, interpolation = cv2.INTER_LINEAR) * mask
            if no_conf == False:
                conf_big = cv2.resize(conf_img, dim, interpolation = cv2.INTER_LINEAR) * mask
            
            if lens.focal_type == 0:
                count0[ptc_y-hw_y:ptc_y+hw_y+1, ptc_x-hw_x:ptc_x+hw_x+1] += mask
                psimg0[ptc_y-hw_y:ptc_y+hw_y+1, ptc_x-hw_x:ptc_x+hw_x+1] = mask * ps#color_img.shape[0] * color_img.shape[1]
                img_lens_type0[ptc_y-hw_y:ptc_y+hw_y+1, ptc_x-hw_x:ptc_x+hw_x+1, 0:3] += img_big[:,:,0:3]
                disp_lens_type0[ptc_y-hw_y:ptc_y+hw_y+1, ptc_x-hw_x:ptc_x+hw_x+1] += disp_big
                if no_conf == False:
                    conf_lens_type0[ptc_y-hw_y:ptc_y+hw_y+1, ptc_x-hw_x:ptc_x+hw_x+1] += conf_big
            elif lens.focal_type == 1:
                count1[ptc_y-hw_y:ptc_y+hw_y+1, ptc_x-hw_x:ptc_x+hw_x+1] += mask
                psimg1[ptc_y-hw_y:ptc_y+hw_y+1, ptc_x-hw_x:ptc_x+hw_x+1] = mask * ps#color_img.shape[0] * color_img.shape[1]
                img_lens_type1[ptc_y-hw_y:ptc_y+hw_y+1, ptc_x-hw_x:ptc_x+hw_x+1, 0:3] += img_big[:,:,0:3]
                disp_lens_type1[ptc_y-hw_y:ptc_y+hw_y+1, ptc_x-hw_x:ptc_x+hw_x+1] += disp_big
                if no_conf == False:
                    conf_lens_type1[ptc_y-hw_y:ptc_y+hw_y+1, ptc_x-hw_x:ptc_x+hw_x+1] += conf_big
            elif lens.focal_type == 2:
                count2[ptc_y-hw_y:ptc_y+hw_y+1, ptc_x-hw_x:ptc_x+hw_x+1] += mask
                psimg2[ptc_y-hw_y:ptc_y+hw_y+1, ptc_x-hw_x:ptc_x+hw_x+1] = mask * ps#color_img.shape[0] * color_img.shape[1]
                img_lens_type2[ptc_y-hw_y:ptc_y+hw_y+1, ptc_x-hw_x:ptc_x+hw_x+1, 0:3] += img_big[:,:,0:3]
                disp_lens_type2[ptc_y-hw_y:ptc_y+hw_y+1, ptc_x-hw_x:ptc_x+hw_x+1] += disp_big
                if no_conf == False:
                    conf_lens_type2[ptc_y-hw_y:ptc_y+hw_y+1, ptc_x-hw_x:ptc_x+hw_x+1] += conf_big
    
    # Here I should average the three images, but first get them right
    # yes, terribly written, but is temporary I hope
    img_lens_type0_fnl = np.ones_like(img_lens_type0)
    img_lens_type1_fnl = np.ones_like(img_lens_type1)
    img_lens_type2_fnl = np.ones_like(img_lens_type2)
    disp_lens_type0_fnl = np.ones_like(disp_lens_type0)
    disp_lens_type1_fnl = np.ones_like(disp_lens_type1)
    disp_lens_type2_fnl = np.ones_like(disp_lens_type2)
    if no_conf == False:
        conf_lens_type0_fnl = np.ones_like(conf_lens_type0)
        conf_lens_type1_fnl = np.ones_like(conf_lens_type1)
        conf_lens_type2_fnl = np.ones_like(conf_lens_type2)
    count0[(count0 == 0)] = 1
    count1[(count1 == 0)] = 1
    count2[(count2 == 0)] = 1

    for j in range(0,3):
        img_lens_type0_fnl[:,:,j] = img_lens_type0[:,:,j] / count0
        img_lens_type1_fnl[:,:,j] = img_lens_type1[:,:,j] / count1
        img_lens_type2_fnl[:,:,j] = img_lens_type2[:,:,j] / count2
    disp_lens_type0_fnl = disp_lens_type0 / count0
    disp_lens_type1_fnl = disp_lens_type1 / count1   
    disp_lens_type2_fnl = disp_lens_type2 / count2   
    if no_conf == False:
        conf_lens_type0_fnl = conf_lens_type0 / count0
        conf_lens_type1_fnl = conf_lens_type1 / count1   
        conf_lens_type2_fnl = conf_lens_type2 / count2 

    img_lens_type0_fnl[np.isnan(img_lens_type0_fnl)] = 0
    img_lens_type1_fnl[np.isnan(img_lens_type1_fnl)] = 0
    img_lens_type2_fnl[np.isnan(img_lens_type2_fnl)] = 0
    disp_lens_type0_fnl[np.isnan(disp_lens_type0_fnl)] = 0 
    disp_lens_type1_fnl[np.isnan(disp_lens_type1_fnl)] = 0 
    disp_lens_type2_fnl[np.isnan(disp_lens_type2_fnl)] = 0 
    if no_conf == False:
        conf_lens_type0_fnl[np.isnan(disp_lens_type0_fnl)] = 0 
        conf_lens_type1_fnl[np.isnan(disp_lens_type1_fnl)] = 0 
        conf_lens_type2_fnl[np.isnan(disp_lens_type2_fnl)] = 0 

    # select disparity
    avg_disp = (disp_lens_type0_fnl + disp_lens_type1_fnl + disp_lens_type2_fnl) / 3 

    # divide areas
    # lens type 0 --> 1 to 3 virtual depth --> disparity > 0.6
    # lens type 1 --> 3 to 4 virtual depth --> 0.6 > disparity > 0.3
    # lens type 2 --> 4 to 100 virtual depth --> disparity < 0.3
    weights = np.zeros((img_lens_type0_fnl.shape[0], img_lens_type0_fnl.shape[1], 4))
    lens_type0_focus_area = avg_disp > 0.6
    lens_type1_focus_area = (avg_disp > 0.3) * (avg_disp < 0.6)
    lens_type2_focus_area = avg_disp < 0.3
    weights[:,:,0] = 0.6 * lens_type0_focus_area + 0.2 * lens_type1_focus_area + 0.1 * lens_type2_focus_area
    weights[:,:,1] = 0.3 * lens_type0_focus_area + 0.6 * lens_type1_focus_area + 0.3 * lens_type2_focus_area
    weights[:,:,2] = 0.1 * lens_type0_focus_area + 0.2 * lens_type1_focus_area + 0.6 * lens_type2_focus_area
    weights[:,:,3] = np.ones_like(weights[:,:,3])

    all_in_focus_image = (img_lens_type0_fnl * np.dstack((weights[:,:,0], weights[:,:,0], weights[:,:,0], weights[:,:,3])) + \
        img_lens_type1_fnl * np.dstack((weights[:,:,1], weights[:,:,1], weights[:,:,1], weights[:,:,3])) + \
        img_lens_type2_fnl * np.dstack((weights[:,:,2], weights[:,:,2], weights[:,:,2], weights[:,:,3])) ) 

    all_in_focus_image[:,:,3] = 1

    final_disp_img = (disp_lens_type0_fnl * weights[:,:,0] + disp_lens_type1_fnl * weights[:,:,1] + disp_lens_type2_fnl * weights[:,:,2] ) 
    if no_conf == False:
        final_conf_img = (conf_lens_type0_fnl * weights[:,:,0] + conf_lens_type1_fnl * weights[:,:,1] + conf_lens_type2_fnl * weights[:,:,2] ) 
    else:
        final_conf_img = np.zeros_like(final_disp_img)
    avg_ps = (psimg0 + psimg1 + psimg2 ) / 3

    # cutting out the sides where there is no information!
    if cutBorders is True:

        paddingToAvoidBorders = int(max_ps + 1)
        all_in_focus_image = all_in_focus_image[paddingToAvoidBorders:all_in_focus_image.shape[0]-paddingToAvoidBorders, paddingToAvoidBorders:all_in_focus_image.shape[1]-paddingToAvoidBorders, :]
        final_disp_img = final_disp_img[paddingToAvoidBorders:final_disp_img.shape[0]-paddingToAvoidBorders, paddingToAvoidBorders:final_disp_img.shape[1]-paddingToAvoidBorders]
        avg_disp = avg_disp[paddingToAvoidBorders:avg_disp.shape[0]-paddingToAvoidBorders, paddingToAvoidBorders:avg_disp.shape[1]-paddingToAvoidBorders]
        avg_ps = avg_ps[paddingToAvoidBorders:avg_ps.shape[0]-paddingToAvoidBorders, paddingToAvoidBorders:avg_ps.shape[1]-paddingToAvoidBorders]
        final_conf_img = final_conf_img[paddingToAvoidBorders:final_conf_img.shape[0]-paddingToAvoidBorders, paddingToAvoidBorders:final_conf_img.shape[1]-paddingToAvoidBorders]

    return all_in_focus_image, avg_disp, final_disp_img, avg_ps, final_conf_img


"""
GENERATE FOCUSED VIEW

Rewritten for better implemetnation

adding filters in the end

April 2019
"""
def generate_view_focused_micro_lenses_v2(lenses, no_conf=False, x_shift=0, y_shift=0, patch_shape=0, cutBorders=True, isReal=True, imgname=None, chosen=3):
   
    # patch shapes
    # they should be related to the microlens image size (these numbers were good for R29)
    triplet = [[12, 5, 7], [10, 7, 9], [8, 11, 13], [6, 13, 15], [4, 15, 17]]
    lens_size = lenses[0,0].diameter
    #triplet = [[12, 5, 7], [10, 7, 9], [8, 11, 13], [6, 23, 25], [4, 51, 55]]
    chosen = 1
    # for the images captured with 25mm objective
    # triplet25mm = [6, 23, 25] # lens diameter 70 pixels

    # for the old images
    # triplet25mm = [6, 13, 13] # lens diameter 40 pixels

    lens_types = 3
    # we set the patch image to be one/sixth of the original, if not otherwise specified
    factor = triplet[chosen][0] # if changing this the final resolution will change
    central_lens = lenses[0,0]
    img_shape = ((central_lens.pcoord) * 2 + 1).astype(int)
    cen = round(central_lens.img.shape[0]/2.0)
    if len(central_lens.col_img.shape) > 1:
        hl, wl, c = central_lens.col_img.shape
    else:
        hl, wl = central_lens.img.shape
        c = 1
    max_ps = np.floor(central_lens.diameter / 2)

    # WE USE 3 DIMENSIONAL STRUCTURES
    # They basically are layer of images (with color channels, or with one value for disparity and confidence). 
    # Each layer has information from one lens type
    rendered_colors = np.zeros((int(img_shape[0]/factor), int(img_shape[1]/factor), c, lens_types)) 
    if c == 4:
        rendered_colors[:,:,3,:] = 1 # alpha channel
    rendered_disps = np.zeros((int(img_shape[0]/factor), int(img_shape[1]/factor), lens_types)) 
    if no_conf == False:
        rendered_confidences = np.zeros((int(img_shape[0]/factor), int(img_shape[1]/factor), lens_types)) 

    counters = np.zeros((int(img_shape[0]/factor), int(img_shape[1]/factor), lens_types)) 
    patch_sizes = np.zeros((int(img_shape[0]/factor), int(img_shape[1]/factor), lens_types)) 

    # actual size of the patches in the rendered image
    actual_size_x = triplet[chosen][1] #15
    actual_size_y = triplet[chosen][2] #round(hl / factor) + 4
    if actual_size_x % 2 == 0:
        actual_size_x += 1
    dim = (actual_size_x, actual_size_y)
    hw_x = int(np.floor(actual_size_x/2))
    hw_y = int(np.floor(actual_size_y/2))
    # create a mask to actual extract eclipses patches
    radius = np.floor(actual_size_y/2)
    x = np.linspace(-1, 1, actual_size_y) * radius
    xx, yy = np.meshgrid(x, x)
    if patch_shape == 0:
        rect_mask = np.ones_like(xx)
        mask = rect_mask[:,2:rect_mask.shape[1]]
        # for the big microlenses
        #mask = rect_mask[:,2:rect_mask.shape[1]-2]
    elif patch_shape == 1:
        circle_mask = np.zeros_like(xx)
        circle_mask[xx**2 + yy**2 < (radius+1)**2] = 1
        mask = rect_mask[:,2:rect_mask.shape[1]]
        # for the big microlenses
        #mask = rect_mask[:,2:rect_mask.shape[1]-2]
    mask4c = np.dstack((mask, mask, mask, mask))


    # loop and create three images!
    for key in lenses:
        
        #pdb.set_trace()
        lens = lenses[key]
        current_img = np.asarray(lenses[key].col_img)
        current_disp = np.asarray(lenses[key].disp_img)
        if no_conf == False:
            current_conf = np.asarray(lenses[key].conf_img)
        ps = get_patch_size_absolute_focused_lenses(current_disp, lens.diameter, isReal)
        #print(ps)
        cen_y, cen_x = int(np.round(lens.pcoord[0])), int(np.floor(lens.pcoord[1]))
        ptc_y, ptc_x = int(cen_y / factor), int(cen_x / factor)
        if min(ptc_y, ptc_x) > max_ps and ptc_y < (rendered_colors.shape[0]-max_ps) and ptc_x < (rendered_colors.shape[1]-max_ps):       
            #pdb.set_trace()
            color_img = current_img[cen-ps+y_shift:cen+ps+1+y_shift, cen-ps+x_shift:cen+ps+1+x_shift] # patch size!
            disp_simg = current_disp[cen-ps+y_shift:cen+ps+1+y_shift, cen-ps+x_shift:cen+ps+1+x_shift]
            if no_conf == False:
                conf_img = current_conf[cen-ps+y_shift:cen+ps+1+y_shift, cen-ps+x_shift:cen+ps+1+x_shift]
            img_big = cv2.resize(color_img, dim, interpolation = cv2.INTER_LINEAR) * mask4c
            disp_big = cv2.resize(disp_simg, dim, interpolation = cv2.INTER_LINEAR) * mask
            if no_conf == False:
                conf_big = cv2.resize(conf_img, dim, interpolation = cv2.INTER_LINEAR) * mask
            
            # using lens.focal_type we fill only one of the three layers each time
            rendered_colors[ptc_y-hw_y:ptc_y+hw_y+1, ptc_x-hw_x:ptc_x+hw_x+1,0:3, lens.focal_type] += img_big[:,:,0:3]
            rendered_disps[ptc_y-hw_y:ptc_y+hw_y+1, ptc_x-hw_x:ptc_x+hw_x+1, lens.focal_type] += disp_big
            counters[ptc_y-hw_y:ptc_y+hw_y+1, ptc_x-hw_x:ptc_x+hw_x+1, lens.focal_type] += mask
            patch_sizes[ptc_y-hw_y:ptc_y+hw_y+1, ptc_x-hw_x:ptc_x+hw_x+1, lens.focal_type] = mask * ps
            if no_conf == False:
                rendered_confidences[ptc_y-hw_y:ptc_y+hw_y+1, ptc_x-hw_x:ptc_x+hw_x+1, lens.focal_type] += conf_big
    
    counters[(counters == 0)] = 1

    #pdb.set_trace()
    # each lens type has to be divided
    for k in range(0,3):
        # each color channel
        for j in range(0,3):
            rendered_colors[:,:,j,k] /= counters[:,:,k]
        rendered_disps[:,:,k] /= counters[:,:,k]
        #rendered_disps[np.isnan(rendered_disps[:,:,k]), k] = 0
        if no_conf == False:
            rendered_confidences[:,:,k] /= counters[:,:,k]
            #rendered_confidences[np.isnan(rendered_confidences[:,:,k]), k] = 0

    initial_disp = np.mean(rendered_disps, axis=2)
    average_patch_sizes = np.mean(patch_sizes, axis=2)

    # divide areas
    # lens type 0 --> 1 to 3 virtual depth --> disparity > 0.6
    # lens type 1 --> 3 to 4 virtual depth --> 0.6 > disparity > 0.3
    # lens type 2 --> 4 to 100 virtual depth --> disparity < 0.3
    weights = np.zeros((initial_disp.shape[0], initial_disp.shape[1], 4))
    lens_type0_focus_area = initial_disp > 0.6
    lens_type1_focus_area = (initial_disp > 0.3) * (initial_disp < 0.6)
    lens_type2_focus_area = initial_disp < 0.3
    #pdb.set_trace()
    # here we need a bette rway
    weights[:,:,0] = 0.625#0.6 * lens_type0_focus_area + 0.2 * lens_type1_focus_area + 0.1 * lens_type2_focus_area
    weights[:,:,1] = 0.05# * lens_type0_focus_area + 0.6 * lens_type1_focus_area + 0.3 * lens_type2_focus_area
    weights[:,:,2] = 0.325# * lens_type0_focus_area + 0.2 * lens_type1_focus_area + 0.6 * lens_type2_focus_area
    weights[:,:,3] = np.ones_like(weights[:,:,3])

    all_in_focus_image = np.zeros_like(rendered_colors[:,:,:,0])
    final_disp_img = np.zeros_like(rendered_disps[:,:,0])
    final_conf_img = np.zeros_like(final_disp_img)

    #pdb.set_trace()
    for s in range(0,3):

        all_in_focus_image += (rendered_colors[:,:,:,s] * np.dstack((weights[:,:,s], weights[:,:,s], weights[:,:,s], weights[:,:,3])))
        final_disp_img += (rendered_disps[:,:,s] * weights[:,:,s])
        if no_conf == False:
            final_conf_img += (rendered_confidences[:,:,s] * weights[:,:,s])

    all_in_focus_image[:,:,3] = 1

    ### FILTERING
    # the color image is filtered with a soft bilateral filter (standard settings)
    window_size = 13
    sigma_distance = 0.75
    sigma_color = 0.5
    print("Processing colored image..")
    all_in_focus_image = filters.bilateral_filter(all_in_focus_image, window_size, sigma_distance, sigma_color)
    print("Processing disparity map..")
    # sigmaSpatial = 3
    # sigmaColor = 3
    #pdb.set_trace()
    processed_disp = filters.median_filter(final_disp_img, window_size)
    #bf = filters.bilateral_filter(final_disp_img, 13, 1)
    #gf = guidedFilter(all_in_focus_image,final_disp_img.astype(np.float32), 13, 0.5)
    #jbf = jointBilateralFilter(all_in_focus_image, final_disp_img.astype(np.float32), 9, sigmaColor, sigmaSpatial)
    #dtf = dtFilter(all_in_focus_image, final_disp_img.astype(np.float32), sigmaSpatial, sigmaColor)

    #processed_disp = filters.replace_wrong_values(final_disp_img, all_in_focus_image, final_conf_img, minDensity = 0.5)
    #processed_disp2 = filters.replace_wrong_values(final_disp_img, all_in_focus_image, final_conf_img, minDensity = 0.75)
    #processed_disp3 = filters.replace_wrong_values(final_disp_img, all_in_focus_image, final_conf_img, minDensity = 0.25)
    # plt.ion()
    # #plt.imshow(final_disp_img)
    # plt.figure(1); plt.imshow(final_disp_img, cmap='jet', vmin=0.25, vmax= 0.75)
    # plt.figure(2); plt.imshow(mf, cmap='jet', vmin=0.25, vmax= 0.75)
    # #plt.figure(3); plt.imshow(bf, cmap='jet', vmin=0.25, vmax= 0.75)
    # #plt.figure(4); plt.imshow(gf, cmap='jet', vmin=0.25, vmax= 0.75)
    # #plt.figure(4); plt.imshow(jbf, cmap='jet', vmin=0.25, vmax= 0.75)
    # plt.figure(5); plt.imshow(dtf, cmap='jet', vmin=0.25, vmax= 0.75)
    # #plt.show()
    # pdb.set_trace()
    # cutting out the sides where there is no information!
    if cutBorders is True:

        paddingToAvoidBorders = int(max_ps + 1)
        all_in_focus_image = all_in_focus_image[paddingToAvoidBorders:all_in_focus_image.shape[0]-paddingToAvoidBorders, paddingToAvoidBorders:all_in_focus_image.shape[1]-paddingToAvoidBorders, :]
        final_disp_img = final_disp_img[paddingToAvoidBorders:final_disp_img.shape[0]-paddingToAvoidBorders, paddingToAvoidBorders:final_disp_img.shape[1]-paddingToAvoidBorders]
        initial_disp = initial_disp[paddingToAvoidBorders:initial_disp.shape[0]-paddingToAvoidBorders, paddingToAvoidBorders:initial_disp.shape[1]-paddingToAvoidBorders]
        average_patch_sizes = average_patch_sizes[paddingToAvoidBorders:average_patch_sizes.shape[0]-paddingToAvoidBorders, paddingToAvoidBorders:average_patch_sizes.shape[1]-paddingToAvoidBorders]
        final_conf_img = final_conf_img[paddingToAvoidBorders:final_conf_img.shape[0]-paddingToAvoidBorders, paddingToAvoidBorders:final_conf_img.shape[1]-paddingToAvoidBorders]
        processed_disp = processed_disp[paddingToAvoidBorders:processed_disp.shape[0]-paddingToAvoidBorders, paddingToAvoidBorders:processed_disp.shape[1]-paddingToAvoidBorders]

    return all_in_focus_image, initial_disp, final_disp_img, average_patch_sizes, final_conf_img, processed_disp

def get_sampling_distance(disp, calib, sam_per_lens):


    disp_in_pixel = disp * calib.lens_diameter
    if disp_in_pixel < 0.001:
        disp_in_pixel = 0.5
    sam_dist = disp_in_pixel / (2 * sam_per_lens)

    return sam_dist



def _hex_focal_type(c):
    
    """
    Calculates the focal type for the three lens hexagonal grid
    """

    focal_type = ((-c[0] % 3) + c[1]) % 3

    return focal_type 

def render_interp_img(imgs, interps, calibs, shiftx, shifty, cut_borders):
    
    img = imgs[0]
    disp = imgs[1]

    data_interp_r = interps[0]
    data_interp_g = interps[1]
    data_interp_b = interps[2]
    disp_interp = interps[3]
    
    # view
    img_shape = np.asarray(img.shape[0:2])
    calib = calibs[0]
    coords = calibs[1]
    local_grid = calibs[2]

    # resolution should be correlated with number of lenses more than 
    # number of pixels
    #pdb.set_trace()
    # sample per lens
    sam_per_lens = 11
    hs = np.floor(sam_per_lens/2).astype(int)
    
    #[ny * sam_per_lens, nx * sam_per_lens]
    #pdb.set_trace()
    reducing_factor = (calib.lens_diameter / sam_per_lens)
    resolution = np.round(img_shape / reducing_factor).astype(int)
    print("raw image is {}x{}, rendered image will be {}x{}".format(img_shape[0], img_shape[1], resolution[0], resolution[1]))
    rnd_img = np.zeros((resolution[0], resolution[1], 3))

    x, y = local_grid.x, local_grid.y
    # if needed for masking
    # xx, yy = local_grid.xx, local_grid.yy
    # mask = np.zeros_like(local_grid.xx)
    # mask[xx**2 + yy**2 < calib.inner_lens_radius**2] = 1
    #plt.ion()   
    for lc in coords:

        # pixel coordinates
        pc = coords[lc]
        #pdb.set_trace()
        #print("pc[0] = {}, pc[1] = {}".format(pc[0], pc[1]))
        # first we need disparity
        disp_at_pc = disp_interp(y+pc[0], x+pc[1])
        # we need a single value for the disparity
        single_val_disp = np.mean(disp_at_pc)
        # disparity controls distance between pixels
        sampling_distance = get_sampling_distance(single_val_disp, calib, sam_per_lens)
        #print("disp is {}, sampling distance is {}".format(single_val_disp * calib.lens_diameter, sampling_distance))
        # sample the image at the correct position
        #pdb.set_trace()
        coords_resized = pc / reducing_factor
        intPCx = np.ceil(coords_resized[1]).astype(int)
        intPCy = np.ceil(coords_resized[0]).astype(int)
        if intPCx > hs and resolution[1] - intPCx > hs and intPCy > hs and resolution[0] - intPCy > hs:
            sampling_pattern = np.arange(-sampling_distance*sam_per_lens/2, sampling_distance*sam_per_lens/2 + sampling_distance, sampling_distance)
            sampling_pattern_x = sampling_pattern + shiftx
            sampling_pattern_y = sampling_pattern + shifty
            patch_values = np.dstack((data_interp_r(sampling_pattern_y+pc[0], sampling_pattern_x+pc[1]),
                data_interp_g(sampling_pattern_y+pc[0], sampling_pattern_x+pc[1]),
                data_interp_b(sampling_pattern_y+pc[0], sampling_pattern_x+pc[1])))
            patch_values = np.clip(patch_values, 0, np.max(patch_values))
            #print("patch_values size {}".format(patch_values.shape))
            interp_patch_r = sinterp.RectBivariateSpline(range(patch_values.shape[0]), range(patch_values.shape[1]), patch_values[:,:,0])
            interp_patch_g = sinterp.RectBivariateSpline(range(patch_values.shape[0]), range(patch_values.shape[1]), patch_values[:,:,1])
            interp_patch_b = sinterp.RectBivariateSpline(range(patch_values.shape[0]), range(patch_values.shape[1]), patch_values[:,:,2])
            sampling_pattern_for_patch_y = np.arange((intPCy-coords_resized[0]), (intPCy-coords_resized[0]+sam_per_lens), 1)
            sampling_pattern_for_patch_x = np.arange((intPCx-coords_resized[1]), (intPCx-coords_resized[1]+sam_per_lens), 1)
            
            r_channel = interp_patch_r(sampling_pattern_for_patch_y, sampling_pattern_for_patch_x)
            g_channel = interp_patch_g(sampling_pattern_for_patch_y, sampling_pattern_for_patch_x)
            b_channel = interp_patch_b(sampling_pattern_for_patch_y, sampling_pattern_for_patch_x)
            rgb_interp_patch_img = np.dstack((r_channel, g_channel, b_channel))
            rgb_interp_patch_img = np.clip(rgb_interp_patch_img, 0, np.max(rgb_interp_patch_img))
            #print("sam patt y {}, x {}".format(sampling_pattern_for_patch_y.shape, sampling_pattern_for_patch_x.shape))
            #print("size {}:{}, {}:{}".format(intPCy-hs,intPCy+hs+1, intPCx-hs,intPCx+hs+1))
            # pdb.set_trace()
            # plt.figure(1)
            # plt.subplot(131)
            # plt.imshow(patch_values)
            # plt.subplot(132)
            # cX = int(round(pc[1]))
            # cY = int(round(pc[0]))
            # plt.imshow(img[cY-15:cY+16, cX-15:cX+16])
            # plt.subplot(133)
            # plt.imshow((rgb_interp_patch_img))
            #pdb.set_trace()
            rnd_img[intPCy-hs:intPCy+hs+1, intPCx-hs:intPCx+hs+1,:] = rgb_interp_patch_img
            # rnd_img[intPCy-hs:intPCy+hs+1, intPCx-hs:intPCx+hs+1,1] = interp_patch_g(sampling_pattern_for_patch_y, sampling_pattern_for_patch_x)
            # rnd_img[intPCy-hs:intPCy+hs+1, intPCx-hs:intPCx+hs+1,2] = interp_patch_b(sampling_pattern_for_patch_y, sampling_pattern_for_patch_x)
    
    rnd_img = np.clip(rnd_img, 0, 1)
    if cut_borders:
        rnd_img = rnd_img[hs:rnd_img.shape[0]-hs, hs:rnd_img.shape[1]-hs,:]
    #plt.ion()
    #plt.imshow((rnd_img))
    #pdb.set_trace()
    #plt.imsave('/data1/palmieri/COLLABORATIONS/Waqas/IMAGES/RAYTRIX/OUTPUT/RTX008/interp2.png', np.clip(rnd_img, 0, 1))
    return rnd_img


def render_interp_img_focused(imgs, interps, calibs, shiftx, shifty, sam_per_lens, cut_borders):
    
    img = imgs[0]
    disp = imgs[1]

    data_interp_r = interps[0]
    data_interp_g = interps[1]
    data_interp_b = interps[2]
    disp_interp = interps[3]
    
    # view
    img_shape = np.asarray(img.shape[0:2])
    calib = calibs[0]
    coords = calibs[1]
    local_grid = calibs[2]

    # resolution should be correlated with number of lenses more than 
    # number of pixels
    #pdb.set_trace()
    # sample per lens
    hs = np.floor(sam_per_lens/2).astype(int)
    lens_types = 3
    #[ny * sam_per_lens, nx * sam_per_lens]
    #pdb.set_trace()
    reducing_factor = (calib.lens_diameter / sam_per_lens) * 2 #* lens_types
    resolution = np.round(img_shape / reducing_factor).astype(int)
    print("raw image is {}x{}, rendered image will be {}x{}".format(img_shape[0], img_shape[1], resolution[0], resolution[1]))
    rnd_img = np.zeros((resolution[0], resolution[1], 3, lens_types))
    rnd_cnt = np.zeros((resolution[0], resolution[1], 3, lens_types))
    rnd_disp = np.zeros((resolution[0], resolution[1], lens_types))
    coarse_d = np.zeros((resolution[0], resolution[1], lens_types))
    x, y = local_grid.x, local_grid.y
    # if needed for masking
    # xx, yy = local_grid.xx, local_grid.yy
    # mask = np.zeros_like(local_grid.xx)
    # mask[xx**2 + yy**2 < calib.inner_lens_radius**2] = 1
    #plt.ion()  
    for lc in coords:

        # pixel coordinates
        pc = coords[lc]

        ft = _hex_focal_type(lc)
        #print("pc[0] = {}, pc[1] = {}".format(pc[0], pc[1]))
        # first we need disparity
        disp_at_pc = disp_interp(y+pc[0], x+pc[1])
        # we need a single value for the disparity
        single_val_disp = np.mean(disp_at_pc)
        # disparity controls distance between pixels
        sampling_distance = get_sampling_distance(single_val_disp, calib, sam_per_lens) * 2
        #print("disp is {}, sampling distance is {}".format(single_val_disp * calib.lens_diameter, sampling_distance))
        # sample the image at the correct position
        #pdb.set_trace()
        coords_resized = pc / reducing_factor
        intPCx = np.ceil(coords_resized[1]).astype(int)
        intPCy = np.ceil(coords_resized[0]).astype(int)
        if intPCx > hs and resolution[1] - intPCx > hs and intPCy > hs and resolution[0] - intPCy > hs:

            #sampling_pattern = np.arange(-sampling_distance*sam_per_lens/2, sampling_distance*sam_per_lens/2 + sampling_distance, sampling_distance)
            sampling_pattern = np.arange(-sampling_distance*np.floor(sam_per_lens/2), sampling_distance*np.floor(sam_per_lens/2) + sampling_distance, sampling_distance)
            sampling_pattern_x = sampling_pattern + shiftx
            sampling_pattern_y = sampling_pattern + shifty
            # extract the patch 
            patch_values = np.dstack((data_interp_r(sampling_pattern_y+pc[0], sampling_pattern_x+pc[1]),
                data_interp_g(sampling_pattern_y+pc[0], sampling_pattern_x+pc[1]),
                data_interp_b(sampling_pattern_y+pc[0], sampling_pattern_x+pc[1])))
            patch_values = np.clip(patch_values, 0, np.max(patch_values))
            #print("patch_values size {}".format(patch_values.shape))
            # interpolate the values
            interp_patch_r = sinterp.RectBivariateSpline(range(patch_values.shape[0]), range(patch_values.shape[1]), patch_values[:,:,0])
            interp_patch_g = sinterp.RectBivariateSpline(range(patch_values.shape[0]), range(patch_values.shape[1]), patch_values[:,:,1])
            interp_patch_b = sinterp.RectBivariateSpline(range(patch_values.shape[0]), range(patch_values.shape[1]), patch_values[:,:,2])
            
            #create the grid for sampling
            sampling_pattern_for_patch_y = np.arange((intPCy-coords_resized[0]), (intPCy-coords_resized[0]+sam_per_lens), 1)
            sampling_pattern_for_patch_x = np.arange((intPCx-coords_resized[1]), (intPCx-coords_resized[1]+sam_per_lens), 1)

            # get the actual values for each channel
            r_channel = interp_patch_r(sampling_pattern_for_patch_y, sampling_pattern_for_patch_x)
            g_channel = interp_patch_g(sampling_pattern_for_patch_y, sampling_pattern_for_patch_x)
            b_channel = interp_patch_b(sampling_pattern_for_patch_y, sampling_pattern_for_patch_x)

            # stack the 3 channels together
            rgb_interp_patch_img = np.dstack((r_channel, g_channel, b_channel))
            rgb_interp_patch_img = np.clip(rgb_interp_patch_img, 0, np.max(rgb_interp_patch_img))

            # fill the images
            rnd_img[intPCy-hs:intPCy+hs+1, intPCx-hs:intPCx+hs+1,:, ft] += rgb_interp_patch_img 
            rnd_cnt[intPCy-hs:intPCy+hs+1, intPCx-hs:intPCx+hs+1, :, ft] += np.ones((rgb_interp_patch_img.shape[0], rgb_interp_patch_img.shape[1], 3)) 
            coarse_d[intPCy-hs:intPCy+hs+1, intPCx-hs:intPCx+hs+1, ft] += np.ones((rgb_interp_patch_img.shape[0], rgb_interp_patch_img.shape[1])) * single_val_disp
            
    img0vals0 = (rnd_cnt[:,:,:,0] == 0).astype(np.uint8)
    img1vals0 = (rnd_cnt[:,:,:,1] == 0).astype(np.uint8)
    img2vals0 = (rnd_cnt[:,:,:,2] == 0).astype(np.uint8)
    rnd0 = rnd_img[:,:,:,0] / (rnd_cnt[:,:,:,0] + img0vals0)
    rnd1 = rnd_img[:,:,:,1] / (rnd_cnt[:,:,:,1] + img1vals0)
    rnd2 = rnd_img[:,:,:,2] / (rnd_cnt[:,:,:,2] + img2vals0)
    coarse_d0 = coarse_d[:,:,0] / (rnd_cnt[:,:,0,0] + img0vals0[:,:,0])
    coarse_d1 = coarse_d[:,:,1] / (rnd_cnt[:,:,0,1] + img1vals0[:,:,0])
    coarse_d2 = coarse_d[:,:,2] / (rnd_cnt[:,:,0,2] + img2vals0[:,:,0])

    coarse_d_tot = (coarse_d0 + coarse_d1 + coarse_d2 ) / lens_types

    #pdb.set_trace()
    #plt.figure()
    filt_size = min(5, np.round(sam_per_lens/5).astype(int))

    # for the R29 dataset
    range_t0 = [0.400, 1]
    range_t1 = [0, 0.200]
    range_t2 = [0.200, 0.400]
    quantization_step = 0.01
    x = np.arange(0, 1, quantization_step)
    y_t0 = filters.smoothstep(x, range_t0[0], range_t0[1], 0.05)
    y_t1 = filters.smoothstep(x, range_t1[0], range_t1[1], 0.05)
    y_t2 = filters.smoothstep(x, range_t2[0], range_t2[1], 0.05)

    weights_t0 = 1/lens_types + y_t0 / 3 * 2 - y_t1 / 3 * 1 - y_t2 /3 * 1
    weights_t1 = 1/lens_types - y_t0 / 3 * 1 + y_t1 / 3 * 2 - y_t2 /3 * 1
    weights_t2 = 1/lens_types - y_t0 / 3 * 1 - y_t1 / 3 * 1 + y_t2 /3 * 2
    
    weights = np.zeros_like(rnd0)
    coarse_d_tot = np.clip(coarse_d_tot, 0, 0.99)
    idisp = np.floor(coarse_d_tot / quantization_step).astype(np.uint8)
    #pdb.set_trace()
    weights[:,:,0] = weights_t0[idisp]
    weights[:,:,1] = weights_t1[idisp]
    weights[:,:,2] = weights_t2[idisp]

    #pdb.set_trace()
    
    #rnd_tot = rnd0f * weights + rnd1f * weights + rnd2f * weights
    weights0_w3c = np.dstack((weights[:,:,0], weights[:,:,0], weights[:,:,0]))
    weights1_w3c = np.dstack((weights[:,:,1], weights[:,:,1], weights[:,:,1]))
    weights2_w3c = np.dstack((weights[:,:,2], weights[:,:,2], weights[:,:,2]))
    rnd_nof = rnd0 * weights0_w3c + rnd1 * weights1_w3c + rnd2 * weights2_w3c

    filt_after = filters.median_filter(rnd_nof, filt_size)
    
    rnd_img_final = np.clip(filt_after, 0, 1)
    padding = hs + np.floor(hs/2).astype(int)
    if cut_borders:
        rnd_img_final = rnd_img_final[padding:rnd_img_final.shape[0]-padding, padding:rnd_img_final.shape[1]-padding,:]
    
    return rnd_img_final, coarse_d_tot


## The interpolated disparity seems to have some small problems
def render_interp_img_and_disp(imgs, interps, calibs, shiftx, shifty, sam_per_lens, cut_borders):
    
    img = imgs[0]
    disp = imgs[1]

    data_interp_r = interps[0]
    data_interp_g = interps[1]
    data_interp_b = interps[2]
    disp_interp = interps[3]
    
    # view
    img_shape = np.asarray(img.shape[0:2])
    calib = calibs[0]
    coords = calibs[1]
    local_grid = calibs[2]

    # resolution should be correlated with number of lenses more than 
    # number of pixels
    #pdb.set_trace()
    # sample per lens
    hs = np.floor(sam_per_lens/2).astype(int)
    lens_types = 3
    #[ny * sam_per_lens, nx * sam_per_lens]
    #pdb.set_trace()
    reducing_factor = (calib.lens_diameter / sam_per_lens) * 2 #* lens_types
    resolution = np.round(img_shape / reducing_factor).astype(int)
    print("raw image is {}x{}, rendered image will be {}x{}".format(img_shape[0], img_shape[1], resolution[0], resolution[1]))
    rnd_img = np.zeros((resolution[0], resolution[1], 3, lens_types))
    rnd_cnt = np.zeros((resolution[0], resolution[1], 3, lens_types))
    rnd_disp = np.zeros((resolution[0], resolution[1], lens_types))
    coarse_d = np.zeros((resolution[0], resolution[1], lens_types))
    x, y = local_grid.x, local_grid.y
    # if needed for masking
    # xx, yy = local_grid.xx, local_grid.yy
    # mask = np.zeros_like(local_grid.xx)
    # mask[xx**2 + yy**2 < calib.inner_lens_radius**2] = 1
    #plt.ion()   
    #pdb.set_trace()
    for lc in coords:

        # pixel coordinates
        pc = coords[lc]
        #pdb.set_trace()
        ft = _hex_focal_type(lc)
        #print("pc[0] = {}, pc[1] = {}".format(pc[0], pc[1]))
        # first we need disparity
        disp_at_pc = disp_interp(y+pc[0], x+pc[1])
        # we need a single value for the disparity
        single_val_disp = np.mean(disp_at_pc)
        # disparity controls distance between pixels
        sampling_distance = get_sampling_distance(single_val_disp, calib, sam_per_lens) * 2
        #print("disp is {}, sampling distance is {}".format(single_val_disp * calib.lens_diameter, sampling_distance))
        # sample the image at the correct position
        #pdb.set_trace()
        coords_resized = pc / reducing_factor
        intPCx = np.ceil(coords_resized[1]).astype(int)
        intPCy = np.ceil(coords_resized[0]).astype(int)
        if intPCx > hs and resolution[1] - intPCx > hs and intPCy > hs and resolution[0] - intPCy > hs:
            #pdb.set_trace()
            sampling_pattern = np.arange(-sampling_distance*sam_per_lens/2, sampling_distance*sam_per_lens/2 + sampling_distance, sampling_distance)
            sampling_pattern_x = sampling_pattern + shiftx
            sampling_pattern_y = sampling_pattern + shifty
            # extract the patch 
            patch_values = np.dstack((data_interp_r(sampling_pattern_y+pc[0], sampling_pattern_x+pc[1]),
                data_interp_g(sampling_pattern_y+pc[0], sampling_pattern_x+pc[1]),
                data_interp_b(sampling_pattern_y+pc[0], sampling_pattern_x+pc[1])))
            patch_values = np.clip(patch_values, 0, np.max(patch_values))
            # disparity
            disp_patch_values = disp_interp(sampling_pattern_y+pc[0], sampling_pattern_x+pc[1])
            disp_patch_values = np.clip(disp_patch_values, 0, np.max(disp_patch_values))
            #print("patch_values size {}".format(patch_values.shape))
            # interpolate the values
            interp_patch_r = sinterp.RectBivariateSpline(range(patch_values.shape[0]), range(patch_values.shape[1]), patch_values[:,:,0])
            interp_patch_g = sinterp.RectBivariateSpline(range(patch_values.shape[0]), range(patch_values.shape[1]), patch_values[:,:,1])
            interp_patch_b = sinterp.RectBivariateSpline(range(patch_values.shape[0]), range(patch_values.shape[1]), patch_values[:,:,2])
            interp_patch_d = sinterp.RectBivariateSpline(range(disp_patch_values.shape[0]), range(disp_patch_values.shape[1]), disp_patch_values[:,:])
            #create the grid for sampling
            sampling_pattern_for_patch_y = np.arange((intPCy-coords_resized[0]), (intPCy-coords_resized[0]+sam_per_lens), 1)
            sampling_pattern_for_patch_x = np.arange((intPCx-coords_resized[1]), (intPCx-coords_resized[1]+sam_per_lens), 1)
            
            # get the actual values for each channel
            r_channel = interp_patch_r(sampling_pattern_for_patch_y, sampling_pattern_for_patch_x)
            g_channel = interp_patch_g(sampling_pattern_for_patch_y, sampling_pattern_for_patch_x)
            b_channel = interp_patch_b(sampling_pattern_for_patch_y, sampling_pattern_for_patch_x)
            d_channel = interp_patch_d(sampling_pattern_for_patch_y, sampling_pattern_for_patch_x)

            # stack the 3 channels together
            rgb_interp_patch_img = np.dstack((r_channel, g_channel, b_channel))

            # clip in  case interpolation gives some values slightly above 1 or below 0
            rgb_interp_patch_img = np.clip(rgb_interp_patch_img, 0, 1) #np.max(rgb_interp_patch_img))
            d_interp_patch_img = np.clip(d_channel, 0, 1) #np.max(d_interp_patch_img))
            
            # fill the images
            rnd_img[intPCy-hs:intPCy+hs+1, intPCx-hs:intPCx+hs+1,:, ft] += rgb_interp_patch_img
            rnd_cnt[intPCy-hs:intPCy+hs+1, intPCx-hs:intPCx+hs+1, :, ft] += np.ones((rgb_interp_patch_img.shape[0], rgb_interp_patch_img.shape[1], 3))
            rnd_disp[intPCy-hs:intPCy+hs+1, intPCx-hs:intPCx+hs+1, ft] += d_interp_patch_img
            coarse_d[intPCy-hs:intPCy+hs+1, intPCx-hs:intPCx+hs+1, ft] += np.ones((rgb_interp_patch_img.shape[0], rgb_interp_patch_img.shape[1])) * single_val_disp
            
    #pdb.set_trace()
    img0vals0 = (rnd_cnt[:,:,:,0] == 0).astype(np.uint8)
    img1vals0 = (rnd_cnt[:,:,:,1] == 0).astype(np.uint8)
    img2vals0 = (rnd_cnt[:,:,:,2] == 0).astype(np.uint8)
    rnd0 = rnd_img[:,:,:,0] / (rnd_cnt[:,:,:,0] + img0vals0)
    rnd1 = rnd_img[:,:,:,1] / (rnd_cnt[:,:,:,1] + img1vals0)
    rnd2 = rnd_img[:,:,:,2] / (rnd_cnt[:,:,:,2] + img2vals0)
    coarse_d0 = coarse_d[:,:,0] / (rnd_cnt[:,:,0,0] + img0vals0[:,:,0])
    coarse_d1 = coarse_d[:,:,1] / (rnd_cnt[:,:,0,1] + img1vals0[:,:,0])
    coarse_d2 = coarse_d[:,:,2] / (rnd_cnt[:,:,0,2] + img2vals0[:,:,0])
    rnd_disp0 = rnd_disp[:,:,0] / (rnd_cnt[:,:,0,0] + img0vals0[:,:,0])
    rnd_disp1 = rnd_disp[:,:,1] / (rnd_cnt[:,:,0,1] + img1vals0[:,:,0])
    rnd_disp2 = rnd_disp[:,:,2] / (rnd_cnt[:,:,0,2] + img2vals0[:,:,0])

    # coarse disp
    coarse_d_tot = (coarse_d0 + coarse_d1 + coarse_d2 ) / lens_types

    #pdb.set_trace()
    #plt.figure()
    filt_size = min(5, np.round(sam_per_lens/5).astype(int))

    # for the R29 dataset
    range_t0 = [0.400, 1]
    range_t1 = [0, 0.200]
    range_t2 = [0.200, 0.400]
    quantization_step = 0.01
    x = np.arange(0, 1, quantization_step)
    y_t0 = filters.smoothstep(x, range_t0[0], range_t0[1], 0.05)
    y_t1 = filters.smoothstep(x, range_t1[0], range_t1[1], 0.05)
    y_t2 = filters.smoothstep(x, range_t2[0], range_t2[1], 0.05)

    weights_t0 = 1/lens_types + y_t0 / 3 * 2 - y_t1 / 3 * 1 - y_t2 /3 * 1
    weights_t1 = 1/lens_types - y_t0 / 3 * 1 + y_t1 / 3 * 2 - y_t2 /3 * 1
    weights_t2 = 1/lens_types - y_t0 / 3 * 1 - y_t1 / 3 * 1 + y_t2 /3 * 2
    
    weights = np.zeros_like(rnd0)
    idisp = np.floor(coarse_d_tot / quantization_step).astype(np.uint8)
    #pdb.set_trace()
    weights[:,:,0] = weights_t0[idisp]
    weights[:,:,1] = weights_t1[idisp]
    weights[:,:,2] = weights_t2[idisp]

    #pdb.set_trace()
    rnd_disp_nof = rnd_disp0 * weights[:,:,0] + rnd_disp1 * weights[:,:,1] + rnd_disp2 * weights[:,:,2]
    
    #rnd_tot = rnd0f * weights + rnd1f * weights + rnd2f * weights
    weights0_w3c = np.dstack((weights[:,:,0], weights[:,:,0], weights[:,:,0]))
    weights1_w3c = np.dstack((weights[:,:,1], weights[:,:,1], weights[:,:,1]))
    weights2_w3c = np.dstack((weights[:,:,2], weights[:,:,2], weights[:,:,2]))
    rnd_nof = rnd0 * weights0_w3c + rnd1 * weights1_w3c + rnd2 * weights2_w3c

    filt_after = filters.median_filter(rnd_nof, filt_size)
    disp_filt = filters.median_filter(rnd_disp_nof, filt_size*2-1)
    #pdb.set_trace()

    rnd_img_final = np.clip(filt_after, 0, 1)
    rnd_disp_final = np.clip(disp_filt, 0, 1)
    padding = hs + np.floor(hs/2).astype(int)
    if cut_borders:
        rnd_img_final = rnd_img_final[padding:rnd_img_final.shape[0]-padding, padding:rnd_img_final.shape[1]-padding,:]
        coarse_d_tot = coarse_d_tot[padding:coarse_d_tot.shape[0]-padding, padding:coarse_d_tot.shape[1]-padding]
        rnd_disp_final = rnd_disp_final[padding:rnd_disp_final.shape[0]-padding, padding:rnd_disp_final.shape[1]-padding]
    #plt.ion()
    #plt.imshow((rnd_img))
    #pdb.set_trace()
    #plt.imsave('/data1/palmieri/COLLABORATIONS/Waqas/IMAGES/RAYTRIX/OUTPUT/RTX008/interp2.png', np.clip(rnd_img, 0, 1))
    return rnd_img_final, coarse_d_tot, rnd_disp_final

"""
Rendering of the images
The idea would be the following:
1 - first take the disparities and calculate a single value 
(if we have a coarse map, that would be ideal)
2 - then we use this value to calculate the patch size that we need
and the sampling distance
3 - then we sample RGB, disparity and confidence, obtaining 3 images.
4 - we use a light filter on RGB and then consider it as guide image.
5 - we filter more heavily the disparity, trying to remove errors using
disparity and trying to correct them using the color information
------------------
This method should be more general and could be called with different
parameters/input
------------------
November/Dezember 2019
"""
def render_SI(imgs, interps, calibs, info, shiftx, shifty, sam_per_lens, cut_borders, alreadyInterpolated=False):
    
    img = imgs[0]
    disp = imgs[1]
    #pdb.set_trace()
    if len(disp.shape) > 2 and disp[:,:,0].all() == disp[:,:,1].all():
        disp = disp[:,:,0]
    # we should have the confidence, but if we don't, we assume all pixels are equals!
    if len(imgs) > 2:
        conf = imgs[2]
        if len(conf.shape) > 2 and conf[:,:,0].all() == conf[:,:,1].all():
            conf = conf[:,:,0]
    else:
        conf = np.ones_like(disp)

    if alreadyInterpolated:
        data_interp_r = interps[0]
        data_interp_g = interps[1]
        data_interp_b = interps[2]
        disp_interp = interps[3]
    
    # view
    img_shape = np.asarray(img.shape[0:2])
    calib = calibs[0]
    coords = calibs[1]
    local_grid = calibs[2]

    # resolution should be correlated with number of lenses more than 
    # number of pixels
    #pdb.set_trace()
    # sample per lens
    hs = np.floor(sam_per_lens/2).astype(int)
    lens_types = 3
    #[ny * sam_per_lens, nx * sam_per_lens]
    #pdb.set_trace()
    reducing_factor = (calib.lens_diameter / sam_per_lens) * 2 #* lens_types
    resolution = np.round(img_shape / reducing_factor).astype(int)
    print("raw image is {}x{}, rendered image will be {}x{}".format(img_shape[0], img_shape[1], resolution[0], resolution[1]))
    rnd_img = np.zeros((resolution[0], resolution[1], 3, lens_types))
    rnd_cnt = np.zeros((resolution[0], resolution[1], 3, lens_types))
    rnd_disp = np.zeros((resolution[0], resolution[1], lens_types))
    coarse_d = np.zeros((resolution[0], resolution[1], lens_types))
    x, y = local_grid.x, local_grid.y
    # if needed for masking
    xx, yy = local_grid.xx, local_grid.yy
    mask = np.zeros_like(local_grid.xx)
    mask[xx**2 + yy**2 < calib.inner_lens_radius**2] = 1
    #plt.ion()   
    #pdb.set_trace()
    confidenceThreshold = 0.3;
    MIN_DISP_STEP = 0.001
    
    half_space = np.floor((x.shape[0] - sam_per_lens) /2).astype(int)
    x_grid_patch = x[half_space:x.shape[0]-half_space]
    y_grid_patch = y[half_space:y.shape[0]-half_space]
    ptc_xx, ptc_yy = np.meshgrid(x_grid_patch,y_grid_patch)
    patch_mask = np.zeros((sam_per_lens, sam_per_lens))
    hsp = np.ceil(sam_per_lens / 2)
    patch_mask[ptc_xx**2 + ptc_yy**2 < hsp**2] = 1
    # mask to get the circular shape of the patch
    # half_space = np.floor((x.shape[0] - (sam_per_lens * 2 + 1) ) /2).astype(int)
    # patch_mask = mask[half_space:mask.shape[0]-half_space, half_space:mask.shape[1]-half_space]

    #plt.ion()

    for lc in coords:

        # pixel coordinates
        pc = coords[lc]
        # pdb.set_trace()
        ft = _hex_focal_type(lc)

        # disp_at_pc = disp_interp(y+pc[0], x+pc[1])
        # # we need a single value for the disparity
        # single_val_disp2 = np.mean(disp_at_pc)
        # we take the disparity image and its corresponding confidence
        _x = np.round(pc[0]).astype(int)
        _y = np.round(pc[1]).astype(int)
        _radius = np.floor((calib.lens_diameter)/2).astype(int)
        #pdb.set_trace()
        disp_square = disp[_x-_radius:_x+_radius+1, _y-_radius:_y+_radius+1]
        conf_masked = conf[_x-_radius:_x+_radius+1, _y-_radius:_y+_radius+1] * mask
        # find the good ones
        indices_good_pixels = np.where(conf_masked > confidenceThreshold)
        #if len(indices_good_pixels) > 0 and len(indices_good_pixels[0]) < (.5 * np.sum(mask)):
        #    indices_good_pixels = np.where(conf_masked > .8 * (np.sum(conf_masked) / np.sum(mask)))
            #print("now there are {} good pixels".format(len(indices_good_pixels[0])))
        # check if they are actually good!
        good_disp_pixels_as_array = np.asarray(disp_square[indices_good_pixels])
        # we need a single value for the disparity
        if len(good_disp_pixels_as_array) < 10:
            single_val_disp = 0.01
        else:
            single_val_disp = np.mean(good_disp_pixels_as_array)
        
        # print("disp interp: {}, disp with array: {}".format(single_val_disp2, single_val_disp))
        # plt.subplot(121); plt.imshow(img[_x-_radius:_x+_radius+1, _y-_radius:_y+_radius+1,:])
        # plt.subplot(122); plt.imshow(disp_square)
        #pdb.set_trace()
        #print("disp was {}".format(single_val_disp))
        if single_val_disp < MIN_DISP_STEP:
            single_val_disp = MIN_DISP_STEP
        patch_size_for_sampling = single_val_disp * info['dmax'] / 1 # * calib.lens_diameter) / 2

        # disparity controls distance between pixels
        #sampling_distance = get_sampling_distance(single_val_disp, calib, sam_per_lens) * 2
        #print("disp is {}, sampling distance is {}".format(single_val_disp * calib.lens_diameter, sampling_distance))
        # sample the image at the correct position
        # pdb.set_trace()
        coords_resized = pc / reducing_factor
        intPCx = np.ceil(coords_resized[1]).astype(int)
        intPCy = np.ceil(coords_resized[0]).astype(int)
        if intPCx > hs and resolution[1] - intPCx > hs and intPCy > hs and resolution[0] - intPCy > hs:
            
            #pdb.set_trace()
            #print("patch size for sampling is {}".format(patch_size_for_sampling))
            sampling_pattern = np.arange(-patch_size_for_sampling, patch_size_for_sampling+0.0001, (2*patch_size_for_sampling) / (sam_per_lens))
            #print("sampling_pattern: {}".format(sampling_pattern))
            sampling_pattern_x = sampling_pattern + shiftx
            sampling_pattern_y = sampling_pattern + shifty
            
            if alreadyInterpolated:

                # extract the patch 
                patch_values = np.dstack((data_interp_r(sampling_pattern_y+pc[0], sampling_pattern_x+pc[1]),
                    data_interp_g(sampling_pattern_y+pc[0], sampling_pattern_x+pc[1]),
                    data_interp_b(sampling_pattern_y+pc[0], sampling_pattern_x+pc[1])))
                patch_values = np.clip(patch_values, 0, np.max(patch_values))
                # disparity
                disp_patch_values = disp_interp(sampling_pattern_y+pc[0], sampling_pattern_x+pc[1])
                disp_patch_values = np.clip(disp_patch_values, 0, np.max(disp_patch_values))
                #print("patch_values size {}".format(patch_values.shape))
            else:
                win_size = np.floor(np.max(sampling_pattern) + 2).astype(int)
                rgb_values = img[_x-win_size:_x+win_size+1, _y-win_size:_y+win_size+1,:]
                data_interp_r = sinterp.RectBivariateSpline(range(rgb_values.shape[0]), range(rgb_values.shape[1]), rgb_values[:,:,0])
                data_interp_g = sinterp.RectBivariateSpline(range(rgb_values.shape[0]), range(rgb_values.shape[1]), rgb_values[:,:,1])
                data_interp_b = sinterp.RectBivariateSpline(range(rgb_values.shape[0]), range(rgb_values.shape[1]), rgb_values[:,:,2])
                disp_interp = sinterp.RectBivariateSpline(range(disp_square.shape[0]), range(disp_square.shape[1]), disp_square * mask)
                # extract the patch 
                # COLOR
                patch_values = np.dstack((data_interp_r(sampling_pattern_y+win_size, sampling_pattern_x+win_size),
                    data_interp_g(sampling_pattern_y+win_size, sampling_pattern_x+win_size),
                    data_interp_b(sampling_pattern_y+win_size, sampling_pattern_x+win_size)))
                patch_values = np.clip(patch_values, 0, np.max(patch_values)) #* np.dstack((patch_mask, patch_mask, patch_mask))
                # disparity
                center_disp_patch = np.asarray(disp_square.shape) / 2
                disp_patch_values = disp_interp(sampling_pattern_y+center_disp_patch[0], sampling_pattern_x+center_disp_patch[1])
                disp_patch_values = np.clip(disp_patch_values, 0, np.max(disp_patch_values)) #* patch_mask

            #print("patch_values size {}".format(patch_values.shape))
            # interpolate the values
            interp_patch_r = sinterp.RectBivariateSpline(range(patch_values.shape[0]), range(patch_values.shape[1]), patch_values[:,:,0])
            interp_patch_g = sinterp.RectBivariateSpline(range(patch_values.shape[0]), range(patch_values.shape[1]), patch_values[:,:,1])
            interp_patch_b = sinterp.RectBivariateSpline(range(patch_values.shape[0]), range(patch_values.shape[1]), patch_values[:,:,2])
            interp_patch_d = sinterp.RectBivariateSpline(range(disp_patch_values.shape[0]), range(disp_patch_values.shape[1]), disp_patch_values[:,:])
            #create the grid for sampling
            sampling_pattern_for_patch_y = np.arange((intPCy-coords_resized[0]), (intPCy-coords_resized[0]+sam_per_lens), 1)
            sampling_pattern_for_patch_x = np.arange((intPCx-coords_resized[1]), (intPCx-coords_resized[1]+sam_per_lens), 1)
            
            # get the actual values for each channel
            r_channel = interp_patch_r(sampling_pattern_for_patch_y, sampling_pattern_for_patch_x)
            g_channel = interp_patch_g(sampling_pattern_for_patch_y, sampling_pattern_for_patch_x)
            b_channel = interp_patch_b(sampling_pattern_for_patch_y, sampling_pattern_for_patch_x)
            d_channel = interp_patch_d(sampling_pattern_for_patch_y, sampling_pattern_for_patch_x)

            # stack the 3 channels together
            rgb_interp_patch_img = np.dstack((r_channel, g_channel, b_channel)) * np.dstack((patch_mask, patch_mask, patch_mask))

            # clip in  case interpolation gives some values slightly above 1 or below 0
            rgb_interp_patch_img = np.clip(rgb_interp_patch_img, 0, 1) #np.max(rgb_interp_patch_img))
            d_interp_patch_img = np.clip(d_channel, 0, 1) * patch_mask#np.max(d_interp_patch_img))

            # fill the images
            rnd_img[intPCy-hs:intPCy+hs+1, intPCx-hs:intPCx+hs+1,:, ft] += rgb_interp_patch_img
            rnd_cnt[intPCy-hs:intPCy+hs+1, intPCx-hs:intPCx+hs+1, :, ft] += np.ones((rgb_interp_patch_img.shape[0], rgb_interp_patch_img.shape[1], 3)) * np.dstack((patch_mask, patch_mask, patch_mask))
            rnd_disp[intPCy-hs:intPCy+hs+1, intPCx-hs:intPCx+hs+1, ft] += d_interp_patch_img
            coarse_d[intPCy-hs:intPCy+hs+1, intPCx-hs:intPCx+hs+1, ft] += np.ones((rgb_interp_patch_img.shape[0], rgb_interp_patch_img.shape[1])) * single_val_disp * patch_mask
          
    img0vals0 = (rnd_cnt[:,:,:,0] == 0).astype(np.uint8)
    img1vals0 = (rnd_cnt[:,:,:,1] == 0).astype(np.uint8)
    img2vals0 = (rnd_cnt[:,:,:,2] == 0).astype(np.uint8)
    rnd0 = rnd_img[:,:,:,0] / (rnd_cnt[:,:,:,0] + img0vals0)
    rnd1 = rnd_img[:,:,:,1] / (rnd_cnt[:,:,:,1] + img1vals0)
    rnd2 = rnd_img[:,:,:,2] / (rnd_cnt[:,:,:,2] + img2vals0)
    coarse_d0 = coarse_d[:,:,0] / (rnd_cnt[:,:,0,0] + img0vals0[:,:,0])
    coarse_d1 = coarse_d[:,:,1] / (rnd_cnt[:,:,0,1] + img1vals0[:,:,0])
    coarse_d2 = coarse_d[:,:,2] / (rnd_cnt[:,:,0,2] + img2vals0[:,:,0])
    rnd_disp0 = rnd_disp[:,:,0] / (rnd_cnt[:,:,0,0] + img0vals0[:,:,0])
    rnd_disp1 = rnd_disp[:,:,1] / (rnd_cnt[:,:,0,1] + img1vals0[:,:,0])
    rnd_disp2 = rnd_disp[:,:,2] / (rnd_cnt[:,:,0,2] + img2vals0[:,:,0])
  
    # coarse disp
    coarse_d_tot = (coarse_d0 + coarse_d1 + coarse_d2 ) / lens_types

    #pdb.set_trace()
    #plt.figure()
    filt_size = min(5, np.round(sam_per_lens/5).astype(int))

    # for the R29 dataset
    range_t0 = [0.400, 1]
    range_t1 = [0, 0.200]
    range_t2 = [0.200, 0.400]
    quantization_step = 0.01
    x = np.arange(0, 1, quantization_step)
    y_t0 = filters.smoothstep(x, range_t0[0], range_t0[1], 0.05)
    y_t1 = filters.smoothstep(x, range_t1[0], range_t1[1], 0.05)
    y_t2 = filters.smoothstep(x, range_t2[0], range_t2[1], 0.05)

    weights_t0 = 1/lens_types + y_t0 / 3 * 2 - y_t1 / 3 * 1 - y_t2 /3 * 1
    weights_t1 = 1/lens_types - y_t0 / 3 * 1 + y_t1 / 3 * 2 - y_t2 /3 * 1
    weights_t2 = 1/lens_types - y_t0 / 3 * 1 - y_t1 / 3 * 1 + y_t2 /3 * 2
    
    weights = np.zeros_like(rnd0)
    idisp = np.floor(coarse_d_tot / quantization_step).astype(np.uint8)
    #pdb.set_trace()
    idisp = np.clip(idisp, 0, 99).astype(int)
    weights[:,:,0] = weights_t0[idisp]
    weights[:,:,1] = weights_t1[idisp]
    weights[:,:,2] = weights_t2[idisp]

    #pdb.set_trace()
    rnd_disp_nof = rnd_disp0 * weights[:,:,0] + rnd_disp1 * weights[:,:,1] + rnd_disp2 * weights[:,:,2]
    
    #rnd_tot = rnd0f * weights + rnd1f * weights + rnd2f * weights
    weights0_w3c = np.dstack((weights[:,:,0], weights[:,:,0], weights[:,:,0]))
    weights1_w3c = np.dstack((weights[:,:,1], weights[:,:,1], weights[:,:,1]))
    weights2_w3c = np.dstack((weights[:,:,2], weights[:,:,2], weights[:,:,2]))
    rnd_nof = rnd0 * weights0_w3c + rnd1 * weights1_w3c + rnd2 * weights2_w3c

    filt_after = filters.median_filter(rnd_nof, filt_size)
    disp_filt = filters.median_filter(rnd_disp_nof, filt_size*2-1)
    #pdb.set_trace()

    rnd_img_final = np.clip(filt_after, 0, 1)
    rnd_disp_final = np.clip(disp_filt, 0, 1)
    padding = hs + np.floor(hs/2).astype(int)
    if cut_borders:
        rnd_img_final = rnd_img_final[padding:rnd_img_final.shape[0]-padding, padding:rnd_img_final.shape[1]-padding,:]
        coarse_d_tot = coarse_d_tot[padding:coarse_d_tot.shape[0]-padding, padding:coarse_d_tot.shape[1]-padding]
        rnd_disp_final = rnd_disp_final[padding:rnd_disp_final.shape[0]-padding, padding:rnd_disp_final.shape[1]-padding]
    # plt.ion()
    # plt.imshow((rnd_img_final))
    # plt.figure()
    # plt.imshow(rnd_disp_final)
    # pdb.set_trace()
    #plt.imsave('/data1/palmieri/COLLABORATIONS/Waqas/IMAGES/RAYTRIX/OUTPUT/RTX008/interp2.png', np.clip(rnd_img, 0, 1))
    return rnd_img_final, coarse_d_tot, rnd_disp_final


def render_interp_img_at_focal_plane(imgs, interps, calibs, focal_plane, sam_per_lens, cut_borders):
    
    img = imgs[0]
    disp = imgs[1]

    data_interp_r = interps[0]
    data_interp_g = interps[1]
    data_interp_b = interps[2]
    disp_interp = interps[3]
    
    # view
    img_shape = np.asarray(img.shape[0:2])
    calib = calibs[0]
    coords = calibs[1]
    local_grid = calibs[2]

    # resolution should be correlated with number of lenses more than 
    # number of pixels
    #pdb.set_trace()
    # sample per lens
    hs = np.floor(sam_per_lens/2).astype(int)
    lens_types = 3
    #[ny * sam_per_lens, nx * sam_per_lens]
    #pdb.set_trace()
    reducing_factor = (calib.lens_diameter / sam_per_lens) * 2 #* lens_types
    resolution = np.round(img_shape / reducing_factor).astype(int)
    print("raw image is {}x{}, rendered image will be {}x{}".format(img_shape[0], img_shape[1], resolution[0], resolution[1]))
    rnd_img = np.zeros((resolution[0], resolution[1], 3, lens_types))
    rnd_cnt = np.zeros((resolution[0], resolution[1], 3, lens_types))
    rnd_disp = np.zeros((resolution[0], resolution[1], lens_types))
    coarse_d = np.zeros((resolution[0], resolution[1], lens_types))
    x, y = local_grid.x, local_grid.y
    # if needed for masking
    # xx, yy = local_grid.xx, local_grid.yy
    # mask = np.zeros_like(local_grid.xx)
    # mask[xx**2 + yy**2 < calib.inner_lens_radius**2] = 1
    #plt.ion()   
    ## the sampling distance is fixed from the focal plane
    #sampling_distance = focal_plane
    patch_size_for_sampling = focal_plane * calib.lens_diameter / 2
    #plt.ion()    
    for lc in coords:

        # pixel coordinates
        pc = coords[lc]
        #pdb.set_trace()
        ft = _hex_focal_type(lc)
        #print("pc[0] = {}, pc[1] = {}".format(pc[0], pc[1]))
        # first we need disparity
        disp_at_pc = disp_interp(y+pc[0], x+pc[1])
        # we need a single value for the disparity
        single_val_disp = np.mean(disp_at_pc)

        #print("disp is {}, sampling distance is {}".format(single_val_disp * calib.lens_diameter, sampling_distance))
        # sample the image at the correct position
        #pdb.set_trace()
        coords_resized = pc / reducing_factor
        intPCx = np.ceil(coords_resized[1]).astype(int)
        intPCy = np.ceil(coords_resized[0]).astype(int)
        if intPCx > sam_per_lens and resolution[1] - intPCx > sam_per_lens and intPCy > sam_per_lens and resolution[0] - intPCy > sam_per_lens:
            #pdb.set_trace()
            #sampling_pattern = np.arange(-sampling_distance*sam_per_lens, sampling_distance*sam_per_lens + sampling_distance, sampling_distance)
            sampling_pattern = np.arange(-patch_size_for_sampling, patch_size_for_sampling, (2 * patch_size_for_sampling) / (2 * sam_per_lens + 1))
            sampling_pattern_x = sampling_pattern 
            sampling_pattern_y = sampling_pattern
            # extract the patch 
            patch_values = np.dstack((data_interp_r(sampling_pattern_y+pc[0], sampling_pattern_x+pc[1]),
                data_interp_g(sampling_pattern_y+pc[0], sampling_pattern_x+pc[1]),
                data_interp_b(sampling_pattern_y+pc[0], sampling_pattern_x+pc[1])))
            patch_values = np.clip(patch_values, 0, np.max(patch_values))
            #print("patch_values size {}".format(patch_values.shape))
            # interpolate the values
            interp_patch_r = sinterp.RectBivariateSpline(range(patch_values.shape[0]), range(patch_values.shape[1]), patch_values[:,:,0])
            interp_patch_g = sinterp.RectBivariateSpline(range(patch_values.shape[0]), range(patch_values.shape[1]), patch_values[:,:,1])
            interp_patch_b = sinterp.RectBivariateSpline(range(patch_values.shape[0]), range(patch_values.shape[1]), patch_values[:,:,2])
            #create the grid for sampling
            #pdb.set_trace()
            sampling_pattern_for_patch_y = np.arange((intPCy-coords_resized[0]), (intPCy-coords_resized[0]+2*sam_per_lens+1), 1)
            sampling_pattern_for_patch_x = np.arange((intPCx-coords_resized[1]), (intPCx-coords_resized[1]+2*sam_per_lens+1), 1)
            
            # get the actual values for each channel
            r_channel = interp_patch_r(sampling_pattern_for_patch_y, sampling_pattern_for_patch_x)
            g_channel = interp_patch_g(sampling_pattern_for_patch_y, sampling_pattern_for_patch_x)
            b_channel = interp_patch_b(sampling_pattern_for_patch_y, sampling_pattern_for_patch_x)
            
            # stack the 3 channels together
            rgb_interp_patch_img = np.dstack((r_channel, g_channel, b_channel))
            rgb_interp_patch_img = np.clip(rgb_interp_patch_img, 0, np.max(rgb_interp_patch_img))
            
            
            
            # plt.figure(1)
            # plt.imshow(rgb_interp_patch_img)
            # plt.figure(2)
            # plt.imshow(img[np.round(pc[0]).astype(np.uint8)-20:np.round(pc[0]).astype(np.uint8)+21, 
            #   np.round(pc[1]).astype(np.uint8)-20:np.round(pc[1]).astype(np.uint8)+21,:])
            # pdb.set_trace()
            hsrgb = np.floor(rgb_interp_patch_img.shape[0]/2).astype(int)
            x = np.linspace(-hsrgb, hsrgb, rgb_interp_patch_img.shape[0])
            y = np.linspace(-hsrgb, hsrgb, rgb_interp_patch_img.shape[1])
            xx, yy = np.meshgrid(x,y)
            mask = np.zeros_like(xx)
            mask[xx**2 + yy**2 < hsrgb**2] = 1  
            mask3c = np.dstack((mask, mask, mask))
            #pdb.set_trace()
            # fill the images
            rnd_img[intPCy-sam_per_lens:intPCy+sam_per_lens+1, intPCx-sam_per_lens:intPCx+sam_per_lens+1,:, ft] += rgb_interp_patch_img * mask3c
            rnd_cnt[intPCy-sam_per_lens:intPCy+sam_per_lens+1, intPCx-sam_per_lens:intPCx+sam_per_lens+1, :, ft] += np.ones((rgb_interp_patch_img.shape[0], rgb_interp_patch_img.shape[1], 3)) * mask3c
            coarse_d[intPCy-sam_per_lens:intPCy+sam_per_lens+1, intPCx-sam_per_lens:intPCx+sam_per_lens+1, ft] += np.ones((rgb_interp_patch_img.shape[0], rgb_interp_patch_img.shape[1])) * single_val_disp * mask
            
    #pdb.set_trace()
    img0vals0 = (rnd_cnt[:,:,:,0] == 0).astype(np.uint8)
    img0vals1 = (rnd_cnt[:,:,:,1] == 0).astype(np.uint8)
    img0vals2 = (rnd_cnt[:,:,:,2] == 0).astype(np.uint8)
    rnd0 = rnd_img[:,:,:,0] / (rnd_cnt[:,:,:,0] + img0vals0)
    rnd1 = rnd_img[:,:,:,1] / (rnd_cnt[:,:,:,1] + img0vals1)
    rnd2 = rnd_img[:,:,:,2] / (rnd_cnt[:,:,:,2] + img0vals2)
    coarse_d0 = coarse_d[:,:,0] / (rnd_cnt[:,:,0,0] + img0vals0[:,:,0])
    coarse_d1 = coarse_d[:,:,1] / (rnd_cnt[:,:,0,1] + img0vals1[:,:,0])
    coarse_d2 = coarse_d[:,:,2] / (rnd_cnt[:,:,0,2] + img0vals2[:,:,0])

    coarse_d_tot = (coarse_d0 + coarse_d1 + coarse_d2 ) / lens_types

    #pdb.set_trace()
    #plt.figure()
    filt_size = min(5, np.round(sam_per_lens/5).astype(int))

    # for the R29 dataset
    range_t0 = [0.400, 1]
    range_t1 = [0, 0.200]
    range_t2 = [0.200, 0.400]
    quantization_step = 0.01
    x = np.arange(0.0, 1.0, quantization_step)
    y_t0 = filters.smoothstep(x, range_t0[0], range_t0[1], 0.05)
    y_t1 = filters.smoothstep(x, range_t1[0], range_t1[1], 0.05)
    y_t2 = filters.smoothstep(x, range_t2[0], range_t2[1], 0.05)

    weights_t0 = 1/lens_types + y_t0 / 3 * 2 - y_t1 / 3 * 1 - y_t2 /3 * 1
    weights_t1 = 1/lens_types - y_t0 / 3 * 1 + y_t1 / 3 * 2 - y_t2 /3 * 1
    weights_t2 = 1/lens_types - y_t0 / 3 * 1 - y_t1 / 3 * 1 + y_t2 /3 * 2
    
    weights = np.zeros_like(rnd0)
    coarse_d_tot = np.clip(coarse_d_tot, 0, 1)
    idisp = np.floor(coarse_d_tot / quantization_step).astype(np.uint8)
    
    
    weights[:,:,0] = weights_t0[idisp]
    weights[:,:,1] = weights_t1[idisp]
    weights[:,:,2] = weights_t2[idisp]

    #pdb.set_trace()
    
    #rnd_tot = rnd0f * weights + rnd1f * weights + rnd2f * weights
    weights0_w3c = np.dstack((weights[:,:,0], weights[:,:,0], weights[:,:,0]))
    weights1_w3c = np.dstack((weights[:,:,1], weights[:,:,1], weights[:,:,1]))
    weights2_w3c = np.dstack((weights[:,:,2], weights[:,:,2], weights[:,:,2]))
    rnd_nof = rnd0 * weights0_w3c + rnd1 * weights1_w3c + rnd2 * weights2_w3c

    filt_after = filters.median_filter(rnd_nof, filt_size)
    #pdb.set_trace()
    #plt.subplot(131); plt.imshow(rnd_tot)
    #plt.subplot(132); plt.imshow(rnd_nof)
    #plt.subplot(133); plt.imshow(filt_after)
    #plt.subplot(131); plt.imshow(filters.median_filter(rnd0, filt_size));
    #plt.subplot(132); plt.imshow(filters.median_filter(rnd1, filt_size));
    #plt.subplot(133); plt.imshow(filters.median_filter(rnd2, filt_size));
    
    rnd_img_final = np.clip(filt_after, 0, 1)
    #plt.imshow(rnd_img_final)
    #pdb.set_trace()
    padding = hs + np.floor(hs/2).astype(int)
    if cut_borders:
        rnd_img_final = rnd_img_final[padding:rnd_img_final.shape[0]-padding, padding:rnd_img_final.shape[1]-padding,:]
    #plt.ion()
    #plt.imshow((rnd_img))
    #pdb.set_trace()
    #plt.imsave('/data1/palmieri/COLLABORATIONS/Waqas/IMAGES/RAYTRIX/OUTPUT/RTX008/interp2.png', np.clip(rnd_img, 0, 1))
    return rnd_img_final, coarse_d_tot

#def focused_view_interp(lenses, no_conf=False, x_shift=0, y_shift=0, patch_shape=0, cutBorders=True, isReal=True, imgname=None, chosen=3):

def createMaskBG(I, bbox):

    #pdb.set_trace()
    # look for blue color
    #BLUE_INTERVAL = [0.3, 0.6, 0.5, 0.8, 0.4, 0.9]
    #BLACK_INTERVAL = [-1, 0.001, -1, 0.001, -1, 0.001]
    maskBG_B = (I[:,:,0] < bbox[1]) * (I[:,:,0] > bbox[0])
    maskBG_G = (I[:,:,1] < bbox[3]) * (I[:,:,1] > bbox[2])
    maskBG_R = (I[:,:,2] < bbox[5]) * (I[:,:,2] > bbox[4])

    mask_RGB = 1 - (maskBG_R * maskBG_G * maskBG_B)
    #if I.shape[2] == 3:
    #    maskBG_RGB = np.dstack((mask_RGB, mask_RGB, mask_RGB))
    #elif I.shape[2] == 4:
    #    maskBG_RGB = np.dstack((mask_RGB, mask_RGB, mask_RGB, np.ones_like(mask_RGB)))
    #else:
    #    print("What? Image has {} channels!".format(I.shape[2]))

    return mask_RGB

def formatAsPCL(image, disparity, scaling):

    bigmesh = np.zeros((image.shape[0] * image.shape[1], 6))
    counter = 0;
    scaling_factor = scaling;
    #pdb.set_trace()
    for i in range(disparity.shape[0]):
        for j in range(disparity.shape[1]):
            #pdb.set_trace()
            if disparity[i,j] > 0 and np.sum(image[i,j,0:3]) > 0:
                bigmesh[counter, 0:3] = [i,j,-disparity[i,j]*scaling_factor]
                bigmesh[counter, 3:6] = (image[i,j,0:3]*255).astype(int)
                counter += 1

    mesh = np.zeros((counter, 6))
    mesh[:,:] = bigmesh[:counter, :]
    return mesh, counter

"""
Just create the header and save as a .ply file for visualization

"""
def save_3D_view(image, disparity, scaling, pcl_directory, pcl_name):

    mesh, counter = formatAsPCL(image, disparity, scaling)

    #pdb.set_trace()
    # saving the ply file
    name_1 = "{}/mesh.txt".format(pcl_directory)
    name_2 = "{}/header.txt".format(pcl_directory)
    name_3 = "{}/{}.ply".format(pcl_directory, pcl_name)
    np.savetxt(name_1, mesh, fmt='%3.3f %3.3f %3.3f %d %d %d')
    print("Saved the 3D View!")
    f = open(name_2, 'w')
    f.write("ply\n\
format ascii 1.0\n\
element vertex ")
    f.write(str(counter))
    f.write("\n\
property float x\n\
property float y\n\
property float z\n\
property uchar red\n\
property uchar green\n\
property uchar blue\n\
element face 0\n\
property list uchar int vertex_index  \n\
end_header\n")
    f.close()
    filenames = [name_2, name_1]
    with open(name_3, 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)

    return 1


def create_mesh_from_disparity(image, disparity, scaling):

    vertex = np.zeros((image.shape[0] * image.shape[1], 6))
    counterV = 0;

    # we use triangles
    faces = np.zeros((image.shape[0] * image.shape[1], 3))
    counterF = 0
    scaling_factor = scaling;

    for i in range(disparity.shape[0]):
        for j in range(disparity.shape[1]):
            #pdb.set_trace()
            if disparity[i,j] > 0 and np.sum(image[i,j,0:3]) > 0:
                vertex[counter, 0:3] = [i,j,-disparity[i,j]*scaling_factor]
                vertex[counter, 3:6] = (image[i,j,0:3]*255).astype(int)
                counter += 1
            # for each pixel create a face
            
              
    mesh = np.zeros((counter, 6))
    mesh[:,:] = bigmesh[:counter, :]
    return mesh, counter
"""
Just create the header and save as a .ply file for visualization

"""
def save_3D_mesh(image, disparity, scaling, pcl_directory, pcl_name):

    vertex, faces, counter_vertex, counter_faces = create_mesh_from_disparity(image, disparity, scaling)

    #pdb.set_trace()
    # saving the ply file
    name_1 = "{}/mesh.txt".format(pcl_directory)
    name_2 = "{}/header.txt".format(pcl_directory)
    name_3 = "{}/{}.ply".format(pcl_directory, pcl_name)
    np.savetxt(name_1, mesh, fmt='%3.3f %3.3f %3.3f %d %d %d')
    print("Saved the 3D View!")
    f = open(name_2, 'w')
    f.write("ply\n\
format ascii 1.0\n\
element vertex ")
    f.write(str(counter_vertex))
    f.write("\n\
property float x\n\
property float y\n\
property float z\n\
property uchar red\n\
property uchar green\n\
property uchar blue\n\
element face ")
    f.write(str(counter_faces))
    f.write("\n\
property list uchar int vertex_index  \n\
end_header\n")
    f.close()
    filenames = [name_2, name_1]
    with open(name_3, 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)

    return 1
