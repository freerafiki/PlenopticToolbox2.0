## FILTERS LIBRARY
import scipy.ndimage.filters
from rendering import render
import microlens.lens as rtxlens
import scipy.interpolate as sinterp
import numpy as np
import cv2
import pdb
import matplotlib.pyplot as plt

def rgb2gray(col_img):

    weights = np.array([0.3, 0.59, 0.11])
    gray = np.sum([col_img[:, :, i].astype(np.float64) * weights[i] for i in range(3)], axis=0)

    return gray

# The scipy ndimage filter adapted for 3-4 channels usage
def median_filter(img, filter_size, footprint=None, changeColorSpace=False):

    if len(img.shape) == 2:

        # normal filter
        filtered_img = scipy.ndimage.filters.median_filter(img, filter_size, footprint)

    elif len(img.shape) == 3:

        # channels
        filtered_img = np.zeros(img.shape)
        for i in range (3):
            filtered_img[:,:,i] = scipy.ndimage.filters.median_filter(img[:,:,i], filter_size, footprint)
        if filtered_img.shape[2] == 4:
            filtered_img[:,:,3] = 1

    else:

        # wrong
        print("is the first argument really an image?")
        filtered_img = np.zeros(img.shape)

    return filtered_img


def bilateral_filter(img, filter_size = 5, sigma_distance = 0.75, sigma_color = 0.5):

    if len(img.shape) == 2:

        # normal filter
        filtered_img = cv2.bilateralFilter(img.astype(np.float32), filter_size, sigma_distance, sigma_color)

    elif len(img.shape) == 3:

        filtered_img = cv2.bilateralFilter(img[:,:,:3].astype(np.float32), filter_size, sigma_distance, sigma_color)

    return filtered_img


def cross_bilateral_filter(img, disp, filter_size = 7, sigma_distance = 1, sigma_color = 0.5, sigma_disp = 0.5):


    return disp

###
# find the threshold to keep minDensity percent pixels of the disparity map
###

def findConfidenceThreshold(confidence_image, minDensity):

    thresh = 0.25
    finished = False
    totalNumberOfPoint = confidence_image.shape[0] * confidence_image.shape[1]
    reducing = False
    increasing = False
    step = 0.1
    finalThresh = 0.0
    iterations = 0
    MAX_ITERATIONS = 20
    while not finished and iterations < MAX_ITERATIONS:
        if reducing == True and increasing == True:
            print("bug in finding the confidence, trying to restart")
            reducing == False
            increasing == False
            thresh = 0.25
            finished = false
        density = np.sum(confidence_image > thresh) / totalNumberOfPoint
        if density < 0 or density > 1:
            print("density is wrong, that cannot be!")
        iterations += 1
        if density > minDensity:
            # reducing density, higher threshold
            if reducing == False and increasing == False:
                reducing = True
            if reducing == False and increasing == True:
                finished = True
                finalThresh = thresh
            thresh += step
            #print("threshold increased to {}".format(thresh))
        elif density < minDensity:
            # increasing density, lower threshold
            if increasing == False and reducing == False:
                increasing = True
            if reducing == True and increasing == False:
                finished = True
                finalThresh = thresh - step
                if finalThresh < 0:
                    finalThresh = 0.1
            thresh -= step
            if thresh < 0:
                thresh = 0.1
                finalThresh = 0.1
                finished = True
            #print("threshold reduced to {}".format(thresh))
        else: #density == minDensity
            finished = True
            finalThresh = thresh

        if finished == True and finalThresh < 0.01:
            print("bug in final threshold. It went to {}. Setting to 0.5".format(finalThresh))
            finalThresh = 0.5 #thresh

    return finalThresh


"""
Filling algorithm

"""
def refillWrongValues(disp, img, wrongValuesMap):

    finished = False
    MAX_ITERATIONS = 10
    iterations = 0
    kernel = np.ones((5,5),np.uint8)
    hw = 6
    weights = np.array([0.3, 0.59, 0.11])
    gray_img = np.sum([img[:, :, i] * weights[i] for i in range(3)], axis=0)
    threshold_gray = np.max(gray_img) / 3
    sigma_gray = np.max(gray_img) / 10
    padding = 20
    filledDisp = np.zeros_like(disp)
    while not finished and iterations < MAX_ITERATIONS:

        #pdb.set_trace()
        #print("Filling disparity.. iter {}".format(iterations+1))
        dilatedVersion = cv2.dilate(wrongValuesMap.astype(np.uint8), kernel, iterations = 3)
        bandToBeFilled = dilatedVersion * wrongValuesMap
        indices = np.where(bandToBeFilled > 0)
        for ind in range(len(indices[0])):
            i = indices[1][ind]
            j = indices[0][ind]
            if i > padding and j > padding and i < img.shape[1] - padding and j < img.shape[1] - padding:
                grayVal = gray_img[j,i]
                #pdb.set_trace()
                # take a window and fill the point
                patchColor = gray_img[j-hw:j+hw+1, i-hw:i+hw+1]
                maskColor = ((patchColor - grayVal) < threshold_gray).astype(int)
                weightsColor = np.exp(-(patchColor - grayVal) / pow(sigma_gray,2))
                patchDisp = disp[j-hw:j+hw+1, i-hw:i+hw+1]
                fillingPatch = maskColor * weightsColor * patchDisp
                fillVal = np.sum(maskColor * weightsColor * patchDisp) / np.sum(maskColor * weightsColor)
                filledDisp[j, i] = fillVal
                wrongValuesMap[j,i] = 0
        if np.sum(filledDisp < 0.001) < 10:
            finished = True
        iterations += 1

    return filledDisp
"""
Idea here is to replace the wrong values (detected using confidence) with neighbouring values
"""
def replace_wrong_values(disp, img, confidence, filter_size = 7, minDensity = 0.75):

    #pdb.set_trace()
    confThreshold = findConfidenceThreshold(confidence, minDensity)
    initialWrongValuesMap = confidence < confThreshold
    # we copy it because it will be modified
    wrongValuesMap = initialWrongValuesMap.copy()
    filledMap = refillWrongValues(disp, img, initialWrongValuesMap)
    new_disp = filledMap * wrongValuesMap + disp * (1-wrongValuesMap)

    return new_disp


def process_disparity_per_lens(lenses, img_shape, method=1, ring=1):

    pdb.set_trace()
    ## METHODS
    # 0 = just improve based on single microlens image filtering using color, depth, confidence..
    # 1 = consistency check only horizontal
    # 2 = improve based on adjacent neighbours, no matter what ring is specified
    # 3 = improve based on neighbours specified from the rings (see below for rings explanation)
    if method == 2:
        ring = 1
    
    enoughPixelRatio = 0.25
    standardMask = getMask(lenses[0,0])
    pixelsInAMicrolensImage = np.sum(standardMask)

    filtered_disp_dict = dict()
    # second idea
    # use neighbouring lenses to improve disparity
    for key in lenses:

        # CHECK THE BORDERS
        lens = lenses[key]
        if min(lens.pcoord[0], lens.pcoord[1]) > 2 * lens.diameter and lens.pcoord[0] < (img_shape[0] - lens.diameter) and lens.pcoord[1] < (img_shape[1] - 2 * lens.diameter):
        
            #
            ### MAYBE CHECK HERE IF LENS IS NOT USEFUL IF MASK IS 0
            if np.sum(lens.mask) > enoughPixelRatio * pixelsInAMicrolensImage:
                pdb.set_trace()
                if method == 0:
                    filtered_disp_dict[key] = improve_disparity_local(lens)
                elif method == 1:
                    lcoordx, lcoordy = lens.lcoord
                    neighbours = [lenses[lcoordx, lcoordy-1], lenses[lcoordx, lcoordy+1]]
                    filtered_disp_dict[key] = improve_disparity_horiz(lens, neighbours)
                elif method == 2:
                    neighbouring_lenses = get_neighbouring_lenses(lenses, lens.lcoord, ring)
                    filtered_disp_dict[key] = improve_disparity(lens, neighbouring_lenses, ring)
                elif method == 3:
                    neighbouring_lenses = get_neighbouring_lenses(lenses, lens.lcoord, ring)
                    filtered_disp_dict[key] = improve_disparity(lens, neighbouring_lenses, ring)
            

    """
    First attempt: 
    find edges in subaperture view and extend them to microimages
    takes time and do not work (the edge finding) so well, so leave it for now

    print("  - Rendering a subaperture image (SI)")
    SI, SIdisp, SIrefdisp, PSimg, SIconf, SIProcDisp = render.generate_view_focused_micro_lenses_v2(lenses)
    print("  - Finding edges in SI ")
    edges = find_edges_SI(SI, SIProcDisp)
    print("  - Getting the edges back to each microlens image (MI) ")
    edges_dict = extend_edges_to_MI(lenses, edges)
    print("  - Filtering MI using edges + confidence information ")
    filtered_disp_dict = filter_MI(lenses, edges_dict)
    """
    return filtered_disp_dict

def improve_disparity_local(lens):
    
    clr = lens.col_img
    disp = lens.disp_img
    conf = lens.conf_img
    
    # if confidence is terrible, let's not use it (consider all 1)
    if np.mean(conf) > 0.35 * np.max(conf):
        conf = np.ones_like(conf)
        
    disp = bilateral_filter(disp)
    
    return disp

def improve_disparity_horiz(lens, neighbours):
    
    #pdb.set_trace()
    clr_L = neighbours[0].col_img
    disp_L = neighbours[0].disp_img
    clr_C = lens.col_img
    disp_C = lens.disp_img
    clr_R = neighbours[1].col_img
    disp_R = neighbours[1].disp_img
    
    pdb.set_trace()
    mask = lens.mask
    initDisp = consistencyCheck(disp_L, disp_C, disp_R, mask)
    filledDisp = freeRefill(initDisp, mask, [disp_L, disp_C, disp_R], [clr_L, clr_C, clr_R])
    finalDisp = median_filter(filledDisp, 5, mask)
    plt.subplot(221); plt.imshow(initDisp * mask); 
    plt.subplot(222); plt.imshow(finalDisp * mask); 
    plt.subplot(223); plt.imshow(clr_C); 
    plt.subplot(224); plt.imshow(mask)
    pdb.set_trace()
    
    return finalDisp

def getMask(lens):

    mask = np.zeros_like(lens.disp_img)
    local_grid = rtxlens.LocalLensGrid(lens.diameter)
    x, y = local_grid.x, local_grid.y
    xx, yy = local_grid.xx, local_grid.yy
    mask = np.zeros_like(local_grid.xx)
    mask[xx**2 + yy**2 < lens.inner_radius**2] = 1  
    
    return mask

def consistencyCheck(dispL, dispC, dispR, mask):
    
    #pdb.set_trace()
    eps = 1
    valid_pixels = np.where(mask > 0)
    gridy, gridx = range(dispC.shape[0]), range(dispC.shape[1])
    dispC_interp = sinterp.RectBivariateSpline(gridy, gridx, dispC)
    dispL_interp = sinterp.RectBivariateSpline(gridy, gridx, dispL)
    dispR_interp = sinterp.RectBivariateSpline(gridy, gridx, dispR)
    for k in range(len(valid_pixels[0])-1):
        i = valid_pixels[0][k]
        j = valid_pixels[1][k]
        c_d = dispC[i,j]
        if abs(c_d) < j:
            if not (abs(dispL_interp(i,j+c_d) - c_d) < eps or mask[i,np.round(j+c_d).astype(int)] < 1) and (abs(dispR_interp(i,j-c_d) - c_d) < eps or mask[i,np.round(j-c_d).astype(int)] < 1):
                dispC[i,j] = 0
  
    return dispC        

def freeRefill(disp, mask, disps, images, hw = 2):
    
    #pdb.set_trace()
    pixelsToBeRefilled = np.where(disp < 0.001)
    weights = np.array([0.3, 0.59, 0.11])
    data = np.sum([images[1][:, :, i] * weights[i] for i in range(3)], axis=0)
    for k in range(len(pixelsToBeRefilled[0])):
        i = pixelsToBeRefilled[0][k]
        j = pixelsToBeRefilled[1][k]  
        if min(i,j) > hw*2+1 and max(i,j) < (disp.shape[0] - hw*2+1):
            #print(i,j)
            #pdb.set_trace()
            weights = abs(data[i-hw:i+hw+1, j-hw:j+hw+1] - np.ones((hw*2+1, hw*2+1)) * data[i,j])
            normalizationFactor = np.sum(weights)
            if np.abs(normalizationFactor) < 0.0001:
                normalizationFactor = 1
            weights /= normalizationFactor
            disp[i,j] = np.sum(disp[i-hw:i+hw+1, j-hw:j+hw+1] * weights)
    
    return disp

def improve_disparity(lens, neighbours, ring):

    
    return 1


"""
Convention for neighbouring lenses

ring = 1 --> a,b,c,d,e,f                            # 6 LENSES
ring = 2 --> g,h,i,j,k,l,m,n,o,p,q,r                # 12 LENSES
ring > 2 --> a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r    # 18 LENSES

          i   j   k
        h   b   c   l
      g   a   0   d   m
        r   f   e   n
          q   p   o

"""
def get_neighbouring_lenses(lensesdict, relative_coordinate, ring):
    
    if ring < 1:
        print("Eh, something wrong, ring < 1! ring = {}".format(ring))

    pdb.set_trace()
    neighbours = dict()
    if ring == 1 or ring > 2:
        neighbours['a'] =  lensesdict[relative_coordinate[0]+0, relative_coordinate[1]-1]
        neighbours['b'] =  lensesdict[relative_coordinate[0]-1, relative_coordinate[1]+1]
        neighbours['c'] =  lensesdict[relative_coordinate[0]-1, relative_coordinate[1]+0]
        neighbours['d'] =  lensesdict[relative_coordinate[0]-0, relative_coordinate[1]+1]
        neighbours['e'] =  lensesdict[relative_coordinate[0]+1, relative_coordinate[1]+0]
        neighbours['f'] =  lensesdict[relative_coordinate[0]+1, relative_coordinate[1]-1]
    if ring > 1:
        neighbours['g'] =  lensesdict[relative_coordinate[0]+0, relative_coordinate[1]-2]
        neighbours['h'] =  lensesdict[relative_coordinate[0]-1, relative_coordinate[1]-1]
        neighbours['i'] =  lensesdict[relative_coordinate[0]-2, relative_coordinate[1]+0]
        neighbours['j'] =  lensesdict[relative_coordinate[0]-2, relative_coordinate[1]+1]
        neighbours['k'] =  lensesdict[relative_coordinate[0]-2, relative_coordinate[1]+2]
        neighbours['l'] =  lensesdict[relative_coordinate[0]-1, relative_coordinate[1]+2]
        neighbours['m'] =  lensesdict[relative_coordinate[0]+0, relative_coordinate[1]+2]
        neighbours['n'] =  lensesdict[relative_coordinate[0]+1, relative_coordinate[1]+1]
        neighbours['o'] =  lensesdict[relative_coordinate[0]+2, relative_coordinate[1]+0]
        neighbours['p'] =  lensesdict[relative_coordinate[0]+2, relative_coordinate[1]-1]
        neighbours['q'] =  lensesdict[relative_coordinate[0]+2, relative_coordinate[1]-2]
        neighbours['r'] =  lensesdict[relative_coordinate[0]+1, relative_coordinate[1]-2]

    return neighbours

"""
Find edges in the rendered image (easier and more precise than in each micro-image)
"""
def find_edges_SI(subaperture_image, subaperture_disp):

    pdb.set_trace()
    edges_color = cv2.Canny(subaperture_image.astype(np.uint8), 100, 200)
    edges_disp = cv2.Canny(subaperture_disp.astype(np.float32), 0.4, 0.8)
    pdb.set_trace()

"""
The idea here is filtering the disparity map using the micro-lens images.
"""



def key_img(I, bbox):
    
    maskBG_B = (I[:,:,0] < bbox[1]) * (I[:,:,0] > bbox[0])
    maskBG_G = (I[:,:,1] < bbox[3]) * (I[:,:,1] > bbox[2])
    maskBG_R = (I[:,:,2] < bbox[5]) * (I[:,:,2] > bbox[4])

    mask_RGB = np.max(0, 1 - (maskBG_R + maskBG_G + maskBG_B))

    return I*np.dstack((mask_RGB, mask_RGB, mask_RGB, mask_RGB)), mask_RGB


def createMaskBG(I):

    #pdb.set_trace()
    # look for blue color
    BLUE_INTERVAL = [0.3, 0.6, 0.5, 0.8, 0.4, 0.9]
    GREEN_INTERVAL = [0.25, 1, 0.4, 1, 0.2, 1] #[110/256, 140/256, 180/256, 210/256, 100/256, 140/256]
    BLACK_INTERVAL = [-1, 0.001, -1, 0.001, -1, 0.001]
    maskBG_B = (I[:,:,0] < GREEN_INTERVAL[1]) * (I[:,:,0] > GREEN_INTERVAL[0])
    maskBG_G = (I[:,:,1] < GREEN_INTERVAL[3]) * (I[:,:,1] > GREEN_INTERVAL[2])
    maskBG_R = (I[:,:,2] < GREEN_INTERVAL[5]) * (I[:,:,2] > GREEN_INTERVAL[4])

    mask_RGB = 1 - (maskBG_R * maskBG_G * maskBG_B)
    #if I.shape[2] == 3:
    #    maskBG_RGB = np.dstack((mask_RGB, mask_RGB, mask_RGB))
    #elif I.shape[2] == 4:
    #    maskBG_RGB = np.dstack((mask_RGB, mask_RGB, mask_RGB, np.ones_like(mask_RGB)))
    #else:
    #    print("What? Image has {} channels!".format(I.shape[2]))

    return mask_RGB

def createMaskBGHSV(I):

    #pdb.set_trace()

    hsv = cv2.cvtColor(I.astype(np.float32), cv2.COLOR_RGB2HSV)
    maxH = np.max(hsv[:,:,0])
    meanH = np.mean(hsv[:,:,0])
    #hh = hsv[:,:,0] < (meanH + 10) * (hsv[:,:,0] > 1)
    #mask_RGB = hh

    hh2 = hsv[:,:,0] > meanH
    mask_RGB = 1 - hh2

    kernel = np.ones((5,5),np.uint8)
    mask_RGB = cv2.erode(mask_RGB.astype(np.uint8),kernel,iterations = 1)

    return mask_RGB

def calculateZeroPlane(disparity):

    #pick the center part
    h, w = disparity.shape
    factor = 4
    stepH = np.round(h/factor).astype(int)
    stepW = np.round(w/factor).astype(int)
    central_part = disparity[stepH:(factor-1)*stepH, stepW:(factor-1)*stepW]

    mean = np.mean(central_part)
    #filtered = median_filter(central_part, 21)
    #meanF = np.mean(filtered)
    #print(mean, meanF)
    #pdb.set_trace()
    
    return mean

def findFilesEndingWith(list, ending): 
    
    imagesList = []
    for path in list:

        if path[-len(ending)::] == ending:
            imagesList.append(path)

    return imagesList

def smoothclamp(x, mi, mx): 

    return (lambda t: np.where(t < 0 , 0, np.where( t <= 1 , 3*t**2-2*t**3, 1 ) ) )( (x-mi)/(mx-mi) )

def smoothstep(x, goingup, goingdown, curvingrange):

    y = np.zeros_like(x)
    ones_indices = np.where((x > (goingup + curvingrange) ) & (x < (goingdown - curvingrange)) )
    y[ones_indices] = 1
    
    curveup = np.where((x >= (goingup - curvingrange) ) & (x <= (goingup + curvingrange)) )
    if goingup < curvingrange:
        y[curveup] = 1
    else:
        y[curveup] = 0.5 + (x[curveup] - goingup  ) / (x[curveup[0][-1]] - x[curveup[0][0]])
    curvedown = np.where((x >= (goingdown - curvingrange) ) & (x <= (goingdown + curvingrange)) )
    if goingdown > x[-1] - curvingrange:
        y[curvedown] = 1
    else:
        y[curvedown] = 0.5 + ( x[curvedown] - goingdown ) / (x[curvedown[0][0]] - x[curvedown[0][-1]])

    return y