"""
Improve disparity using focal stack
The disparity computed through matching is noisy sometimes. We use the focal stack and the output of the rendering (image and disparity) to compute an improved version 

THE PIPELINE

- read image and disparity
- compute one view and its disparity and confidence
- read focal stack
- preprocessing: filter on each colored image - activate with --pre
- cost volume computation: using a focus measure - choose with -focus 'focus_measure'
                                                focus_measures: 'tenengrad', 'DoG', 'DoL' 
- cost volume filtering: apply filter on each cost volume slice - choose with -filt 'filter'
                                                                filters: 'median', 'guided', 'bilateral'
- cost volume refinement apply sgm using the rendered view as guide
- compute confidence
- disparity label extraction: choose method - choose with -disp 'method'
                                            methods: 'graph_cut', 'taylor'
- final post-processing: apply a median filter - activate with --post

-------
March 2020
"""

import matplotlib.pyplot as plt 
import numpy as np 
import pdb 
import scipy.ndimage as ndimage
import scipy.signal as signal
import argparse, json, os
# internal imports
import rendering.filters as filt
import disparity.sgm as rtxsgm
import disparity.disparity_calculation as rtxdisp
import plenopticIO.imgIO as xmlio
import rendering.render as rtxrnd
#import pywt



def calculate_focus_map(img, focus_measure, ref_img=None, filt_size = 5):

    #pdb.set_trace()
    ## TENENGRAD VARIANCE, from Pertuz et al. (2013) - Matlab Code ported
    if focus_measure == 'tenengrad':
        Gx, Gy = filt.sobel_separate(img)
        G = np.sum(np.power(Gx,2), axis=2) + np.sum(np.power(Gy,2), axis=2)
        fm = G / 100
    ## Difference of Gaussians
    elif focus_measure == 'DoG':
        fm = 1-filt.DoG(img, 0.1, 3)
    elif focus_measure == 'laplacian':
        laplacian_img = ndimage.filters.laplace(filt.rgb2gray(img))
        mean_lap = np.mean(laplacian_img)
        fm = np.power((laplacian_img - mean_lap),2)
        fm = 1 - fm / np.max(fm)
    elif focus_measure == 'gvar':
        gray = filt.rgb2gray(img)
        meaned = gray - np.mean(gray)
        windowed = signal.convolve2d(meaned, np.ones((7,7)), mode='same')
        fm = np.power(windowed, 2)
        fm = 1 - fm / np.max(fm)
    elif focus_measure == 'wav':
        pdb.set_trace()
        coeffs = pywt.dwt2(img, 'db6')
        LL, (LH, HL, HH) = coeffs
        fm = LL[:,:,0]
    elif focus_measure == 'diff':
        if ref_img is not None:
            diff = np.sum(np.abs(ref_img - img), axis=2)
            windowed = signal.convolve2d(diff, np.ones((filt_size, filt_size)), mode='same')
            fm = windowed #np.power(windowed, 2)
        else:
            raise Exception("no reference image passed!")
    else:
        fm = np.ones_like(img[:,:,0])
    return fm

def extract_disparity(cost_volume, extraction_method, f_planes):

    if extraction_method == 'taylor':
        disparities_interp, disparities_val = rtxdisp.cost_minima_interp(cost_volume, f_planes)

    #pdb.set_trace()
    return disparities_interp

def filter_cost_volume_slice(slice, method, kernel_size):

    if method == 'median':
        filtered_slice = filt.median_filter(slice, kernel_size)
    elif method == 'bilateral':
        filtered_slice = filt.median_filter(slice, kernel_size)
    elif method == 'none':
        filtered_slice = slice

    return filtered_slice

def est_conf_map(cost_volume, sigma, method='mlm'):

    minimum_costs = np.min(cost_volume, axis=2)
    exp_cost = np.exp(-minimum_costs/(2*np.power(conf_sigma,2)))
    denom_cost = np.sum(np.exp(-cost_volume/(2*np.power(conf_sigma,2))), axis=2)
    confidence_map = exp_cost / denom_cost
    confidence_map[np.isnan(confidence_map)] = 0

    return confidence_map

def merge_maps(disp1, disp2, conf_d2, good_pixels_d2, debug=False):

    largest_diff = (np.max(disp1) - np.min(disp1))
    tolerance = largest_diff / 10
    maps_agree = np.abs(disp1 - disp2) < tolerance
    maps_disagree = 1 - maps_agree
    maps_disagree_with_d2 = maps_disagree * good_pixels_d2
    maps_alone = 1 - maps_agree - maps_disagree_with_d2

    # ok now, different weights 
    weights_in_agreement = maps_agree * 0.5
    weights_in_disagreement = 1 - conf_d2# 1 - (np.abs(disp2-disp1) / largest_diff) 
    weights_alone = maps_alone
    
    # sum together
    disp_in_agreement = maps_agree * (weights_in_agreement * disp1 + (1- weights_in_agreement) * disp2)
    disp_in_disagreement = maps_disagree_with_d2 * (weights_in_disagreement * disp1 + (1- weights_in_disagreement) * disp2)
    disp_alone = maps_alone * (weights_alone * disp1 + (1- weights_alone) * disp2) # last part is useless, but good for understanding

    zero_check = disp_in_disagreement * disp_in_agreement * disp_alone
    if np.sum(zero_check) > 0.1:
        print("areas are overlapping - not good!")

    final_map_full = disp_in_disagreement + disp_in_agreement + disp_alone
    final_map_sparse = disp_in_disagreement + disp_in_agreement

    if debug:
        plt.subplot(231)
        plt.imshow(disp1)
        plt.title("Disp 1 - SM")
        plt.subplot(232)
        plt.imshow(disp2)
        plt.title("Disp 1 - DfF")
        plt.subplot(234)
        plt.imshow(disp_in_disagreement)
        plt.title("Areas where they disagree")
        plt.subplot(235)
        plt.imshow(disp_in_agreement)
        plt.title("Areas where they agree")
        plt.subplot(236)
        plt.imshow(disp_alone)
        plt.title("Areas where only D1 has values")
        plt.subplot(233)
        plt.imshow(final_map_full)
        plt.title("Merged Disp")
        plt.show()
        pdb.set_trace()

    return final_map_full, final_map_sparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Read an image and create a focal stack")
    parser.add_argument(dest='input_filename', nargs=1, help="parameters json file")
    parser.add_argument('-o', dest='output_path', default=None)
    parser.add_argument('-fpmin', dest='fp_min', default=None)
    parser.add_argument('-fpmax', dest='fp_max', default=None)
    parser.add_argument('-fstep', dest='fp_step', default=None)
    # LOOK AT THE BEGINNING OF THE FILES FOR EXPLANATION ABOUT PIPELINE and PARAMETERS
    parser.add_argument('-focus', dest='focus_measure', default='DoG')
    parser.add_argument('-ffw', dest='focus_filter_window_size', default='5')
    parser.add_argument('-extr', dest='disp_extraction', default='taylor')
    parser.add_argument('-filt', dest='cv_filtering', default='median')
    parser.add_argument('--pre', dest='pre_processing', default=False, action='store_true')
    parser.add_argument('--post', dest='post_processing', default=False, action='store_true')
    parser.add_argument('--sgm', dest='sgm', default=False, action='store_true')
    parser.add_argument('-col', dest='colorimage_path', default=None)
    parser.add_argument('-conf', dest='conf_path', default=None)
    parser.add_argument('-disp', dest='disp_path', default=None)
    parser.add_argument('-cfg', dest='config_path', default=None)
    parser.add_argument('-json', dest='parent_json', default=None)
    parser.add_argument('-numd', dest='number_of_disparities', default=12)
    parser.add_argument('--borders', default=True, action='store_false')
    parser.add_argument('--savecolor', dest='save_with_colors', default=False, action='store_true')
    parser.add_argument('--show', dest='showResults', default=False, action='store_true')

    args = parser.parse_args()

    input_file = args.input_filename[0]
    parent_json = ""
    if not os.path.exists(input_file):
        print("\nERROR MESSAGE:")
        print('Sorry we need the parameters file (.json). Theres is no file at the path you gave. Please put full path')
        print('It should have been saved from the render_focal_stack.py script')
        print('Please give the path when running this file (ex. python3 disparity_v2.py path_of_the_file.json)')
        print('If you did not create the focal stack, create it by launching the script render_focal_stack.py\n')
        raise OSError("Stopped! Read error message")
    # or the parameters.json file is provided, then we can read the missing part from there
    else:
        print("\nConfiguration file found correctly!")
        with open(input_file) as f:
            parameters = json.load(f)
            if args.fp_min is None:
                args.fp_min = parameters['fp_min']
            if args.fp_max is None:
                args.fp_max = parameters['fp_max']
            if args.fp_step is None:
                args.fp_step = parameters['fp_step']
            num_planes = parameters['planes_num']
            planes = np.asarray(parameters['planes'])
            directory = parameters['directory']
            paths = parameters['paths']
            w = parameters['width']
            h = parameters['height']
            sample_per_lens = parameters['sample_per_lens']
            if args.parent_json is None:
                parent_json = parameters['parent_json']
            else:
                parent_json = args.parent_json
        info = dict()
        with open(parent_json) as f:
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
            # store a couple of infos
            info['dmin'] = parameters['dmin']
            info['dmax'] = parameters['dmax']
            info['confidence_measure'] = parameters['conf']
            info['stereo_measure'] = parameters['technique']
            info['picture_name'] = parameters['pic_name']
    
    print("\n*********************")
    print("********STEP1********")
    print("*********************")
    print("Reading the raw image and per lens disparity..")
    images_paths = [args.colorimage_path, args.disp_path, args.conf_path]
    usingInterpolationAtFirstStep = False
    images, calibs, interps = xmlio.load_files(args.config_path, images_paths, usingInterpolationAtFirstStep)
    print("Generating central view and disparity..")
    view, coarse_disp, fine_disp = rtxrnd.render_SI(images, interps, calibs, info, 0, 0, sample_per_lens, args.borders, usingInterpolationAtFirstStep)
    print("*********************\n")

    # PARAMETERS
    ### IMPORTANT : if this parameters are not set correctly, 
    ###             it could be that a lot of things makes no sense
    ###             penalty1 and penalty2 are for sgm
    ###             conf_sigma is for the confidence (a wrong value will bring all to zero)
    median_filter_kernel_size = 7
    penalty1 = 0.001
    penalty2 = 0.002
    max_cost = 1
    conf_sigma = 1.5
    # we accept values with confidence > mean_conf - std_allow * std_conf
    # so higher values, accept more, lower values, accept less (sparser disparity map)
    std_allow = 0.1 
    #
    filt_size = int(args.focus_filter_window_size)

    # STRUCTUREs
    cost_volume = np.zeros((h, w, num_planes))
    image_stack = np.zeros((h, w, 3, num_planes))

    # PRINT OUT SETUP
    print("\n*********************")
    print("********STEP2********")
    print("*********************")
    if args.pre_processing:
        print("Pre-processing on rgb images enabled!")
    else:
        print("Pre-processing on rgb images disabled!")
    print("Focus measure chosen:", args.focus_measure)
    print("Cost volume uses:", args.cv_filtering)
    print("Disp extraction is done through:", args.disp_extraction)
    if args.sgm:
        print("Semi-global matching step on cost volume enabled!")
    else:
        print("Semi-global matching step on cost volume disabled!")
    if args.post_processing:
        print("Post-processing on disparity map enabled!")
    else:
        print("Post-processing on disparity map disabled!")
    

    #LOOP
    print("Reading images and calculating initial cost volume..")
    for i in range(num_planes):

        cur_img = plt.imread(paths[i])
        cur_img = cur_img[:,:,:3]
        if args.pre_processing:
            cur_img = filt.median_filter(cur_img, median_filter_kernel_size)
        image_stack[:,:,:3,i] = cur_img
        cost_volume_slice = calculate_focus_map(cur_img, args.focus_measure, view, filt_size)
        cost_volume[:,:,i] = filter_cost_volume_slice(cost_volume_slice, args.cv_filtering, median_filter_kernel_size)
        #pdb.set_trace()
        
    # COST REFINEMENT
    if args.sgm:
        print("Applying semi-global matching to refine cost volume..")
        F = np.flipud(np.rot90(cost_volume.T))
        #pdb.set_trace()
        # the regularized cost volume
        imgI = filt.rgb2gray(view)
        mask = np.ones_like(imgI)
        cost_volume = rtxsgm.sgm(imgI, cost_volume, mask, penalty1, penalty2, False, max_cost)

    # CONFIDENCE
    confidence_map = est_conf_map(cost_volume, conf_sigma)
    good_pixels_map = confidence_map > (np.mean(confidence_map) - std_allow * np.std(confidence_map))

    # DISPARITY EXTRACTION
    print("Extracting disparity..")

    # here we want to allineate disparities, so we use the patch size as common factor
    # RENDERING FROM DISP: ps = disp * info['dmax']
    # RENDERING FOCAL STACK: ps = fp * calib.lens_diameter / 2
    # ps = ps --> disp = (fp * calib.lens_diameter) / (2 * info['dmax'])
    # there is a 0.5 factor somewhere that I lost
    # calib is calibs[0]
    disp_at_focal_planes = .5 * (planes * calibs[0].lens_diameter) / ( 2 * info['dmax'])
    disparity_map = extract_disparity(cost_volume, args.disp_extraction, disp_at_focal_planes)
    if args.post_processing:
        disparity_map = filt.median_filter(disparity_map, median_filter_kernel_size)

    print("\n*********************")
    print("********STEP3********")
    print("*********************")
    print("What happens if we fuse information from the two disparities?")
    # Here we fuse the two of them - actually we just add the information of the 
    # focus-based disparity map and try to smooth out the outcome
    combined_disp_map_full, combined_disp_map_sparse = merge_maps(fine_disp, disparity_map, confidence_map, good_pixels_map)


    dfs_directory = args.output_path + '/DispFromFocalStack' + '_spl' + str(sample_per_lens) + '/'
    print("Creating an output folder at", dfs_directory)
    if not os.path.exists(dfs_directory):
        os.makedirs(dfs_directory)
    
    print("Saving there..")
    plt.imsave(dfs_directory + "/img.png", view)
    plt.imsave(dfs_directory + "/matching_disp.png", fine_disp, cmap='gray')
    plt.imsave(dfs_directory + "/focus_disp.png", disparity_map * good_pixels_map, cmap='gray')
    plt.imsave(dfs_directory + "/combined_disp.png", combined_disp_map_full, cmap='gray')
    plt.imsave(dfs_directory + "/combined_disp_sparse.png", combined_disp_map_sparse, cmap='gray')
    if args.save_with_colors:
        print("Saving also a coloured version..")
        plt.imsave(dfs_directory + "/matching_disp_col.png", fine_disp, cmap='jet')
        plt.imsave(dfs_directory + "/focus_disp_col.png", disparity_map * good_pixels_map, cmap='jet')
        plt.imsave(dfs_directory + "/combined_disp_col.png", combined_disp_map_full, cmap='jet')
        plt.imsave(dfs_directory + "/combined_disp_sparse_col.png", combined_disp_map_sparse, cmap='jet')

    print("Done!")
    #plt.imshow(disparity_map, cmap='jet')
    #plt.show()
    print("*********************\n")
    
    if args.showResults:
        plt.subplot(221)
        plt.imshow(view)
        plt.title('RGB Image')
        plt.subplot(222)
        plt.imshow(fine_disp, cmap='jet', vmin=0, vmax=1)
        plt.title('SM Disp')
        plt.subplot(223)
        plt.imshow(disparity_map * good_pixels_map, cmap='jet', vmin=0, vmax=1)
        plt.title('DfF disparity')
        plt.subplot(224)
        plt.imshow(combined_disp_map_full , cmap='jet', vmin=0, vmax=1)
        plt.title('COMBINED disparity')
        plt.show()




