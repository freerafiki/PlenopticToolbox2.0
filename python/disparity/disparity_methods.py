"""
The main architecture of the disparity estimation algorithm is here:
the structure containing the microimage is used to calculate per-image disparity map
the final images are returned
----
@veresion v1.1 - Januar 2017
@author Luca Palmieri
"""
import disparity.disparity_calculation as rtxdisp
import plenopticIO.lens_grid as rtxhexgrid
import rendering.render as rtxrender
import plenopticIO.imgIO as rtxIO
import disparity.sgm as rtxsgm
import numpy as np
import argparse
import math
import os
import json
import pdb
import cv2


def estimate_disp(args):

    B = np.array([[np.sqrt(3)/2, 0.5], [0, 1]]).T

    rings = [int(i) for i in args.use_rings.split(',')]
    
    nb_offsets = []
    for i in rings: 
        nb_offsets.extend(rtxhexgrid.HEX_OFFSETS[i])
    
    lenses, scene_type = load_scene(args.filename)

    diam = lenses[0, 0].diameter
    max_disp = float(args.max_disp) 
    min_disp = float(args.min_disp) 
    num_disp = float(args.num_disp)
    
    #max_lens_dist = np.linalg.norm(np.dot(B, rtxhexgrid.HEX_OFFSETS[args.max_ring][0]))
    
    disparities = np.arange(min_disp, max_disp, (max_disp - min_disp) / num_disp) #16.0 / max_lens_dist)
    #disparities = np.arange(min_disp, 0.7 * diam, 4.0 / max_lens_dist)
    print("Disparities: {0}".format(disparities))
    
    strategy_args = dict()
    selection_strategy = None
    
    if args.method == 'plain':
        fine_costs, coarse_costs, coarse_costs_merged, lens_variance, num_comparisons  = calc_costs_plain(lenses, disparities, nb_offsets, args.max_cost, technique)
    elif args.method == 'real_lut':
        strategy_args = dict()
        strategy_args['target_lenses'] = _precalc_angular()
        strategy_args['min_disp'] = min_disp
        strategy_args['max_disp'] = max_disp
        strategy_args['trade_off'] = args.lut_trade_off
        selection_strategy = real_lut
        
    if selection_strategy is not None:
        print("Selection strategy: {0}".format(selection_strategy))
        fine_costs, coarse_costs, coarse_costs_merged, lens_variance, num_comparisons, disp_avg = calc_costs_selective_with_lut(lenses, disparities, selection_strategy, args.technique, nb_args=strategy_args, refine=args.refine, max_cost=args.max_cost)

    if args.coarse is True:
        coarse_disp = regularize_coarse(lenses, coarse_costs_merged, disparities, penalty1=args.coarse_penalty1, penalty2=args.coarse_penalty2)
        fine_costs = augment_costs_coarse(fine_costs, coarse_disp, lens_variance, disparities, coarse_weight=args.coarse_weight, struct_var=args.struct_var)
     
    fine_disps, fine_disps_interp, fine_val, wta_depths, wta_depths_interp, wta_val, confidence = regularized_fine(lenses, fine_costs, disparities, args.penalty1, args.penalty2, args.max_cost, conf_sigma=args.conf_sigma)
       
    Dsgm = rtxrender.render_lens_imgs(lenses, fine_disps_interp)
    Dwta = rtxrender.render_lens_imgs(lenses, wta_depths_interp)
    
    lens_data = dict()
    col_data = dict()
    if scene_type == 'synth':
        gt_disp = dict()
        
    for lcoord in lenses:
        lens_data[lcoord] = lenses[lcoord].img
        col_data[lcoord] = lenses[lcoord].col_img
        if scene_type == 'synth':
            gt_disp[lcoord] = lenses[lcoord].disp_img
        
    I = rtxrender.render_lens_imgs(lenses, lens_data)
    Dconf = rtxrender.render_lens_imgs(lenses, confidence)
    Icol = rtxrender.render_lens_imgs(lenses, col_data)
    #pdb.set_trace()
    new_offset = [lenses[0,0].pcoord[0] - (Icol.shape[0]/2), lenses[0,0].pcoord[1] - (Icol.shape[1]/2)]

    Dcoarse = None
    if args.coarse is True:
        Dcoarse = rtxrender.render_lens_imgs(lenses, coarse_disp)
        
    if scene_type == 'synth':
        Dgt = rtxrender.render_lens_imgs(lenses, gt_disp)
        error_measurements = analyze_disp(lenses, fine_disps_interp, True)
    else:
        Dgt = None
        sgm_err = None
        wta_err = None
        sgm_err_mask = None
        sgm_err_mse = None
        err_img_r = None
        img_s = None

    return Icol, Dsgm, Dwta, Dgt, Dconf, Dcoarse, disparities, num_comparisons, disp_avg, new_offset, error_measurements

def _has_neighbours(lens, lenses, neighbours):
        
    for l in neighbours:
        if tuple(l + lens.lcoord) not in lenses:
            return False
    
    return True

def get_depth_discontinuities(lenses):
    """
    It computes via Canny (opencv implementation) the discontinuities on the ground truth
    It returns two dictionaries with image masks (one discontinuities, one smooth areas)
    After Canny dilation is used to get more consistent border (ideally 3 pixels large edges)
    ---
    January 2018
    """
    disc = dict()
    smooth = dict()
    for key in lenses:
        current_disp = lenses[key].disp_img
        norm_disp = current_disp / np.max(current_disp)
        int_disp = np.uint8(norm_disp * 255)
        canny = cv2.Canny(int_disp, 100, 200)
        kernel = np.ones((3,3),np.uint8)
        dilation = cv2.dilate(canny,kernel,iterations = 1)
        disc[key] = dilation / 255
        smooth[key] = (255 - dilation) / 255
    
    return disc, smooth
    
def analyze_disp(lenses, est_depths, depth_discontinuities=False, max_ring=5):

    """
    Used only on synthetic images
    Loop through the estimated depth and calculate the following error measurements:
    - average error
    - mean squared error (MSE)
    - standard deviation
    - BadPix 1.0 and 2.0 (% of pixels with error higher than 1% or 2%
    - (Error on depth discontinuities if the third parameter is True)
    """
    #pdb.set_trace()
    err_avg = {0: np.array([]), 1: np.array([]), 2: np.array([])}
    err_mask = {0: np.array([]), 1: np.array([]), 2: np.array([])}
    err_mse = {0: np.array([]), 1: np.array([]), 2: np.array([])}
    bump = {0: np.array([]), 1: np.array([]), 2: np.array([])}
    bump_thresh = 0.25
    badPix1 = np.zeros((len(est_depths)))
    badPix2 = np.zeros((len(est_depths)))
    badPixGraph = np.zeros((len(est_depths), 21))
    badpixindex = 0
    
    # evaluate error on depth discontinuities
    badPix1Disc = np.zeros((len(est_depths)))
    badPix1Smooth = np.zeros((len(est_depths)))
    badPix2Disc = np.zeros((len(est_depths)))
    badPix2Smooth = np.zeros((len(est_depths)))
    badPixGraphDisc = np.zeros((len(est_depths), 21))
    badPixGraphSmooth = np.zeros((len(est_depths), 21))
    disc, smooth = get_depth_discontinuities(lenses)
    avgErrDisc = {0: np.array([]), 1: np.array([]), 2: np.array([])}
    avgErrSmooth = {0: np.array([]), 1: np.array([]), 2: np.array([])}
    
    nb_tmp = []
    for ring in range(max_ring):
        nb_tmp.extend(rtxhexgrid.HEX_OFFSETS[ring])
        
    #pdb.set_trace()    
    for lcoord in est_depths:
        if _has_neighbours(lenses[lcoord], lenses, nb_tmp) is False:
            continue
    
        est = est_depths[lcoord]
        gt = lenses[lcoord].disp_img
        ind = est > 0
        #pdb.set_trace()
        ft = lenses[lcoord].focal_type
        lens = lenses[lcoord]
        mask = ((lens.grid.xx**2 + lens.grid.yy**2) <= lens.inner_radius**2)       
        err_avg[ft] = np.append(err_avg[ft], np.ravel(np.abs(est[mask] - gt[mask])))
        #pdb.set_trace()
        abs_diff = np.ravel(np.abs(est[mask] - gt[mask]))
        err_mask[ft] = np.append(err_mask[ft], abs_diff)
        bumpy = np.clip(abs_diff, 0, bump_thresh)
        bump[ft] = bumpy
        err_mse[ft] = np.append(err_mse[ft], np.ravel( (est[mask] - gt[mask])**2 ))
        err_img_cur = np.abs(est - gt)
        # mask
        to_zero = ((lens.grid.xx**2 + lens.grid.yy**2) > lens.inner_radius**2)
        err_img_cur[to_zero] = 0
        badPix1[badpixindex] += len(np.where(err_img_cur > 1)[0])
        badPix2[badpixindex] += len(np.where(err_img_cur > 2)[0])
        #pdb.set_trace()
        for i in range(0,21):
            badPixGraph[badpixindex, i] += len(np.where(err_img_cur > 0.1*i)[0])
        
        if depth_discontinuities:
            #pdb.set_trace()
            disc_err = err_img_cur[disc[lcoord] > 0.5]
            smth_err = err_img_cur[disc[lcoord] < 0.5]
            #print(len(disc_err), len(smth_err), len(np.ravel( (est[mask] - gt[mask])**2 )))
            avgErrDisc[ft] = np.append(avgErrDisc[ft], disc_err)
            avgErrSmooth[ft] = np.append(avgErrSmooth[ft], smth_err)
            badPix1Disc[badpixindex] += len(np.where(disc_err > 1)[0])
            badPix1Smooth[badpixindex] += len(np.where(smth_err > 1)[0])
            badPix2Disc[badpixindex] += len(np.where(disc_err > 2)[0])
            badPix2Smooth[badpixindex] += len(np.where(smth_err > 2)[0])
            for i in range(0,21):
                badPixGraphDisc[badpixindex, i] += len(np.where(disc_err > 0.1*i)[0])
                badPixGraphSmooth[badpixindex, i] += len(np.where(smth_err > 0.1*i)[0])
        badpixindex += 1    
        
    #pdb.set_trace()    
    final_err = dict()
    final_err_mask = dict()
    final_err_mse = dict()
    bumpiness = dict()
    depth_disc = dict() 
    depth_smooth = dict()
    
    for key in err_avg:
    
        #pdb.set_trace()
        final_err[key] = dict()
        final_err[key]['err'] = np.mean(err_avg[key])
        #print(final_err[key]['err'])
        final_err[key]['std'] = np.std(err_avg[key])
        final_err[key]['max'] = np.max(err_avg[key])
        final_err[key]['num'] = np.mean(err_avg[key] >= (final_err[key]['err'] + 2*final_err[key]['std']))   
         


        final_err_mask[key] = dict()
        final_err_mask[key]['err'] = np.mean(err_mask[key])
        #print(final_err[key]['err'])
        final_err_mask[key]['std'] = np.std(err_mask[key])
        final_err_mask[key]['max'] = np.max(err_mask[key])
        final_err_mask[key]['num'] = np.mean(err_mask[key] >= (final_err_mask[key]['err'] + 2*final_err_mask[key]['std']))



        final_err_mse[key] = dict()
        final_err_mse[key]['err'] = np.mean(err_mse[key])
        #print(final_err[key]['err'])
        final_err_mse[key]['std'] = np.std(err_mse[key])
        final_err_mse[key]['max'] = np.max(err_mse[key])
        final_err_mse[key]['num'] = np.mean(err_mse[key] >= (final_err_mse[key]['err'] + 2*final_err_mse[key]['std']))
    
    

        bumpiness[key] = dict()
        bumpiness[key]['err'] = np.mean(bump[key])
        #print(final_err[key]['err'])
        bumpiness[key]['std'] = np.std(bump[key])
        bumpiness[key]['max'] = np.max(bump[key])
        bumpiness[key]['num'] = np.mean(bump[key] >= (bumpiness[key]['err'] + 2*bumpiness[key]['std']))  
    
       
        if depth_discontinuities:
 
            depth_disc[key] = dict()
            depth_disc[key]['err'] = np.mean(avgErrDisc[key])
            depth_disc[key]['std'] = np.std(avgErrDisc[key])
            depth_disc[key]['max'] = np.max(avgErrDisc[key])
            depth_smooth[key] = dict()
            depth_smooth[key]['err'] = np.mean(avgErrSmooth[key])
            depth_smooth[key]['std'] = np.std(avgErrSmooth[key])
            depth_smooth[key]['max'] = np.max(avgErrSmooth[key])
        
    return final_err, final_err_mask, final_err_mse, [badPix1, badPix2, badPixGraph], bumpiness, [depth_disc, depth_smooth, badPix1Disc, badPix2Disc, badPix1Smooth, badPix2Smooth, badPixGraphDisc, badPixGraphSmooth]

def load_scene(filename):

    basename, suffix = os.path.splitext(filename)

    if suffix == '.json':
          lenses = rtxIO.load_from_json(filename)
          scene_type = 'synth'
    elif suffix == '.xml':
        img_filename = basename + '.png'
        lenses = rtxIO.load_from_xml(img_filename, filename)
        scene_type = 'real'

    return lenses, scene_type
    
def _rel_to_abs(lcoord, lenses, offsets):

    """
    Generate the axial coordinates for the lens lcoord from the given nb offsets
    """
    
    elements = [lenses.get((lcoord[0] + d[0], lcoord[1] + d[1])) for d in offsets]
    return [x for x in elements if x is not None]
    
def _precalc_angular():
    
    # hex basis
    B = np.array([[np.sqrt(3)/2, 0.5], [0, 1]]).T

    # next 6 neighbours
    ring1 = rtxhexgrid.HEX_OFFSETS[1]
    
    eps = 0.0001
    
    l = dict()
    
    for src in ring1:
        # directional vector in the rectangular coordinate system
        v = np.dot(B, src)
        v = v / np.linalg.norm(v)
        l[tuple(src)] = []

        # cosine of the angle between w an v
        for i, ring in enumerate(reversed(rtxhexgrid.HEX_OFFSETS)):
            if i == 1:
                continue
            tmp = []
            for dst in ring:
                w = np.dot(B, dst)
                w = w / np.linalg.norm(w)

                # k = cosine of the angle between w and v
                k = np.dot(v, w)

                # use only lenses within the correct sector
                if k < 0 or k < np.cos(np.pi/6.0):
                    continue

                tmp.append(dst)
                
            l[tuple(src)].append(tmp)

    return l

def real_lut(lens, lenses, coarse_costs, disparities, max_cost=10.0, nb_args=None):
    
    # get some parameters
    target_lenses = list(nb_args['target_lenses'].values())
    min_disp = nb_args['min_disp']
    max_disp = nb_args['max_disp']
    trade_off = nb_args['trade_off']
    
    
    # LUT to pick best combination or best performances
    B = np.array([[np.sqrt(3)/2, 0.5], [0, 1]]).T
    
    #assert len(coarse_costs) <= len(target_lenses)
    
    offsets = []

    tref = rtxhexgrid.hex_focal_type(lens.lcoord)
    
    #read the lut    
    lut_filename = '../disparity/lut_table.json'
    with open(lut_filename, 'r') as f:
        lut_str = json.load(f)
    
    lut_length = len(lut_str['most_acc_0'])
    lut_step = (max_disp - min_disp) / lut_length
    mavg = 0
    mvavg = 0
    counter = 0
    
    for i, ctmp in enumerate(coarse_costs):
        
        m, mval = rtxdisp.cost_minimum_interp(ctmp, disparities)
        
        mvavg += mval
        mavg += m
        counter += 1
    
    # m avg should be the value for the disparity    
    mavg /= (counter)
    mvavg /= counter
    
    # look for the correct index
    disp_int = lut_str['disp_int_interp']
    disp_int['0'][0] = 20.0 #disparities[len(disparities)-1]
    disp_int[str(len(disp_int)-1)][1] = 0.0
    disp_vals = lut_str['disp_vals_interp']
    found = False
    finished = False
    jj = 0
    while (not found and not finished):
        if jj >= len(disp_int):
            finished = True
            jj = 0
        elif mavg < disp_int[str(jj)][0] and mavg > disp_int[str(jj)][1]:
            found = True
        else:
            jj += 1
    
    ### need to show somehow if I didn't find it
    #if (finished and not found):
        #print("Not found! \nmavg={0}\ndisp_int={1}\ndisp_vals={2}\n".format(mavg, disp_int, disp_vals))
    index_lut = jj #lut_length - math.floor(m / (lut_step)) 
    #print("m:{0}, mval:{1}, index:{2}".format(mavg, mvavg, jj))
    
    if trade_off == 1:
        if tref == 0:
            strat = lut_str['most_acc_0'][index_lut]    
        elif tref == 1:
            strat = lut_str['most_acc_1'][index_lut]
        elif tref == 2:
            strat = lut_str['most_acc_2'][index_lut]
    elif trade_off == 0:
        if tref == 0:
            strat = lut_str['best_perf_0'][index_lut]    
        elif tref == 1:
            strat = lut_str['best_perf_1'][index_lut]
        elif tref == 2:
            strat = lut_str['best_perf_2'][index_lut]
            
    targets = from_strat_to_offsets(strat)
        
    return targets , mavg

def from_strat_to_offsets(strat):

    if strat == 'f1':
        selection_strategy = fixed_selection_strategy_1()
    elif strat == 'f2':
        selection_strategy = fixed_selection_strategy_2()
    elif strat == 'f3':
        selection_strategy = fixed_selection_strategy_3()
    elif strat == 'f4':
        selection_strategy = fixed_selection_strategy_4()
    elif strat == 'f5':
        selection_strategy = fixed_selection_strategy_5()
    elif strat == 'f6':
        selection_strategy = fixed_selection_strategy_6()
    elif strat == 'f7':
        selection_strategy = fixed_selection_strategy_7()
    elif strat == 'f8':
        selection_strategy = fixed_selection_strategy_8()  
    elif strat == 'f9':
        selection_strategy = fixed_selection_strategy_9()  
    elif strat == 'f10':
        selection_strategy = fixed_selection_strategy_10()
    elif strat == 'f11':
        selection_strategy = fixed_selection_strategy_11()
    elif strat == 'f12':
        selection_strategy = fixed_selection_strategy_12()
    elif strat == 'f13':
        selection_strategy = fixed_selection_strategy_13()
    elif strat == 'f14':
        selection_strategy = fixed_selection_strategy_14()
        
    return selection_strategy

"""
The strategies from 1 to 15 are fixed strategies, used only for experimental purposes:
Here they are copied as fixed_strategies (1-8) in order to be able to use them without passing any parameters
For other purposes this part of the code is absolutely useless
"""
def fixed_selection_strategy_1():
    
    """

    """
    
    nb_offsets = dict()
    
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[0])):
        o2 = tuple(rtxhexgrid.HEX_OFFSETS[0][i])
        if not o2 in nb_offsets:
            nb_offsets[tuple(o2)] = np.array(o2)
            
    return [offset for offset in nb_offsets]
    
def fixed_selection_strategy_2():
    raytrix
    """

    """
    
    nb_offsets = dict()
    
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[0])):
        o2 = tuple(rtxhexgrid.HEX_OFFSETS[0][i])
        if not o2 in nb_offsets:
            nb_offsets[tuple(o2)] = np.array(o2)
    
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[4])):
        o0 = tuple(rtxhexgrid.HEX_OFFSETS[4][i])
        if not o0 in nb_offsets:
            nb_offsets[tuple(o0)] = np.array(o0) 
            
    return [offset for offset in nb_offsets]
    
def fixed_selection_strategy_3():
    
    """

    """
    
    nb_offsets = dict()
    
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[0])):
        o2 = tuple(rtxhexgrid.HEX_OFFSETS[0][i])
        if not o2 in nb_offsets:
            nb_offsets[tuple(o2)] = np.array(o2)
    
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[5])):
        o0 = tuple(rtxhexgrid.HEX_OFFSETS[5][i])
        if not o0 in nb_offsets:
            nb_offsets[tuple(o0)] = np.array(o0) 
            
            
    return [offset for offset in nb_offsets]
    
def fixed_selection_strategy_4():
    
    """

    """
    
    nb_offsets = dict()
    
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[4])):
        o2 = tuple(rtxhexgrid.HEX_OFFSETS[4][i])
        if not o2 in nb_offsets:
            nb_offsets[tuple(o2)] = np.array(o2)
    
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[5])):
        o0 = tuple(rtxhexgrid.HEX_OFFSETS[5][i])
        if not o0 in nb_offsets:
            nb_offsets[tuple(o0)] = np.array(o0) 
            
            
    return [offset for offset in nb_offsets]

def fixed_selection_strategy_5():
    
    """

    """
    
    nb_offsets = dict()
    
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[6])):
        o0 = tuple(rtxhexgrid.HEX_OFFSETS[6][i])
        if not o0 in nb_offsets:
            nb_offsets[tuple(o0)] = np.array(o0) 
            
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[7])):
        o1 = tuple(rtxhexgrid.HEX_OFFSETS[7][i])
        if not o1 in nb_offsets:
            nb_offsets[tuple(o1)] = np.array(o1)
            
    return [offset for offset in nb_offsets]
    
def fixed_selection_strategy_6():
    
    """

    """
    
    nb_offsets = dict()
    
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[0])):
        o2 = tuple(rtxhexgrid.HEX_OFFSETS[0][i])
        if not o2 in nb_offsets:
            nb_offsets[tuple(o2)] = np.array(o2)
    
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[6])):
        o0 = tuple(rtxhexgrid.HEX_OFFSETS[6][i])
        if not o0 in nb_offsets:
            nb_offsets[tuple(o0)] = np.array(o0) 
            
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[7])):
        o1 = tuple(rtxhexgrid.HEX_OFFSETS[7][i])
        if not o1 in nb_offsets:
            nb_offsets[tuple(o1)] = np.array(o1)
                        
    return [offset for offset in nb_offsets]
    
def fixed_selection_strategy_7():
    
    """

    """
    
    nb_offsets = dict()
    
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[0])):
        o0 = tuple(rtxhexgrid.HEX_OFFSETS[0][i])
        if not o0 in nb_offsets:
            nb_offsets[tuple(o0)] = np.array(o0) 
            
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[4])):
        o1 = tuple(rtxhexgrid.HEX_OFFSETS[4][i])
        if not o1 in nb_offsets:
            nb_offsets[tuple(o1)] = np.array(o1)
            
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[5])):
        o2 = tuple(rtxhexgrid.HEX_OFFSETS[5][i])
        if not o2 in nb_offsets:
            nb_offsets[tuple(o2)] = np.array(o2)
    
    return [offset for offset in nb_offsets]        
            
def fixed_selection_strategy_8():
    
    """

    """
    
    nb_offsets = dict()
    
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[0])):
        o0 = tuple(rtxhexgrid.HEX_OFFSETS[0][i])
        if not o0 in nb_offsets:
            nb_offsets[tuple(o0)] = np.array(o0) 
            
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[1])):
        o1 = tuple(rtxhexgrid.HEX_OFFSETS[1][i])
        if not o1 in nb_offsets:
            nb_offsets[tuple(o1)] = np.array(o1)
            
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[2])):
        o2 = tuple(rtxhexgrid.HEX_OFFSETS[2][i])
        if not o2 in nb_offsets:
            nb_offsets[tuple(o2)] = np.array(o2)
    
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[3])):
        o3 = tuple(rtxhexgrid.HEX_OFFSETS[3][i])
        if not o3 in nb_offsets:
            nb_offsets[tuple(o3)] = np.array(o3) 
            
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[4])):
        o4 = tuple(rtxhexgrid.HEX_OFFSETS[4][i])
        if not o4 in nb_offsets:
            nb_offsets[tuple(o4)] = np.array(o4)
            
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[5])):
        o5 = tuple(rtxhexgrid.HEX_OFFSETS[5][i])
        if not o5 in nb_offsets:
            nb_offsets[tuple(o5)] = np.array(o5)
                
    return [offset for offset in nb_offsets]       
 
def fixed_selection_strategy_9():

    """

    """
    
    nb_offsets = dict()
            
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[1])):
        o1 = tuple(rtxhexgrid.HEX_OFFSETS[1][i])
        if not o1 in nb_offsets:
            nb_offsets[tuple(o1)] = np.array(o1)
                
    return [offset for offset in nb_offsets]  
    
def fixed_selection_strategy_10():

    """
raytrix
    """
    
    nb_offsets = dict()
    
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[0])):
        o0 = tuple(rtxhexgrid.HEX_OFFSETS[0][i])
        if not o0 in nb_offsets:
            nb_offsets[tuple(o0)] = np.array(o0) 
            
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[1])):
        o1 = tuple(rtxhexgrid.HEX_OFFSETS[1][i])
        if not o1 in nb_offsets:
            nb_offsets[tuple(o1)] = np.array(o1)
                
    return [offset for offset in nb_offsets]  
    
def fixed_selection_strategy_11():

    """

    """
    
    nb_offsets = dict()
            
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[1])):
        o1 = tuple(rtxhexgrid.HEX_OFFSETS[1][i])
        if not o1 in nb_offsets:
            nb_offsets[tuple(o1)] = np.array(o1)

    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[4])):
        o4 = tuple(rtxhexgrid.HEX_OFFSETS[4][i])
        if not o4 in nb_offsets:
            nb_offsets[tuple(o4)] = np.array(o4)
            
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[5])):
        o5 = tuple(rtxhexgrid.HEX_OFFSETS[5][i])
        if not o5 in nb_offsets:
            nb_offsets[tuple(o5)] = np.array(o5)
                
    return [offset for offset in nb_offsets]  

def fixed_selection_strategy_12():

    """

    """
    
    nb_offsets = dict()
            
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[1])):
        o1 = tuple(rtxhexgrid.HEX_OFFSETS[1][i])
        if not o1 in nb_offsets:
            nb_offsets[tuple(o1)] = np.array(o1)
            
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[2])):
        o2 = tuple(rtxhexgrid.HEX_OFFSETS[2][i])
        if not o2 in nb_offsets:
            nb_offsets[tuple(o2)] = np.array(o2)
    
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[3])):
        o3 = tuple(rtxhexgrid.HEX_OFFSETS[3][i])
        if not o3 in nb_offsets:
            nb_offsets[tuple(o3)] = np.array(o3) 

    return [offset for offset in nb_offsets]  

def fixed_selection_strategy_13():

    """

    """
    
    nb_offsets = dict()
    
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[0])):
        o0 = tuple(rtxhexgrid.HEX_OFFSETS[0][i])
        if not o0 in nb_offsets:
            nb_offsets[tuple(o0)] = np.array(o0) 
            
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[1])):
        o1 = tuple(rtxhexgrid.HEX_OFFSETS[1][i])
        if not o1 in nb_offsets:
            nb_offsets[tuple(o1)] = np.array(o1)
            
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[4])):
        o4 = tuple(rtxhexgrid.HEX_OFFSETS[4][i])
        if not o4 in nb_offsets:
            nb_offsets[tuple(o4)] = np.array(o4)
            
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[5])):
        o5 = tuple(rtxhexgrid.HEX_OFFSETS[5][i])
        if not o5 in nb_offsets:
            nb_offsets[tuple(o5)] = np.array(o5)
                
    return [offset for offset in nb_offsets]  

def fixed_selection_strategy_14():

    """

    """
    
    nb_offsets = dict()
            
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[2])):
        o2 = tuple(rtxhexgrid.HEX_OFFSETS[2][i])
        if not o2 in nb_offsets:
            nb_offsets[tuple(o2)] = np.array(o2)
    
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[3])):
        o3 = tuple(rtxhexgrid.HEX_OFFSETS[3][i])
        if not o3 in nb_offsets:
            nb_offsets[tuple(o3)] = np.array(o3) 
            
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[4])):
        o4 = tuple(rtxhexgrid.HEX_OFFSETS[4][i])
        if not o4 in nb_offsets:
            nb_offsets[tuple(o4)] = np.array(o4)
            
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[5])):
        o5 = tuple(rtxhexgrid.HEX_OFFSETS[5][i])
        if not o5 in nb_offsets:
            nb_offsets[tuple(o5)] = np.array(o5)
                
    return [offset for offset in nb_offsets]  

def calc_costs_plain(lenses, disparities, nb_offsets, max_cost, progress_hook=print):
    
    coarse_costs = dict()
    coarse_costs_merged = dict()
    fine_costs = dict()
    lens_variance = dict()

    num_lenses = len(lenses)
    num_comparisons = 0
    
    for i, lcoord in enumerate(lenses):
        nb_lenses = _rel_to_abs(lcoord, lenses, nb_offsets)
        lens = lenses[lcoord]
        
        if i%5000==0:
            progress_hook("Processing lens {0}/{1} Coord: {2}".format(i, num_lenses, lcoord))

        fine, coarse, coarse_merged, lens_var = calc_costs_per_lens(lens, nb_lenses, disparities,  max_cost, method)
        
        coarse_costs_merged[lcoord] = coarse_merged
        coarse_costs[lcoord] = coarse
        fine_costs[lcoord], _ = np.array(rtxdisp.merge_costs_additive(fine, max_cost))
        lens_variance[lcoord] = lens_var
        num_comparisons += len(fine)
    
        rtxdisp.assign_last_valid(fine_costs[lcoord])
    return fine_costs, coarse_costs, coarse_costs_merged, lens_variance, num_comparisons

def regularized_fine(lenses, fine_costs, disp, penalty1, penalty2, max_cost, conf_sigma=0.3, min_thresh=2.0, eps=0.0000001):

    fine_depths = dict()
    fine_depths_interp = dict()
    fine_depths_val = dict()
    wta_depths = dict()
    wta_depths_interp = dict()
    wta_depths_val = dict()

    confidence = dict()
    
    for i, l in enumerate(fine_costs):
        
        #pdb.set_trace()
        if i%1000==0:
            print("Processing lens {0}".format(i))
        lens = lenses[l]
        
        # prepare the cost shape: disparity axis is third axis (index [2] instead of [0])
        F = np.flipud(np.rot90(fine_costs[l].T))

        # the regularized cost volume
        sgm_cost = rtxsgm.sgm(lenses[l].img, F, lens.mask, penalty1, penalty2, False, max_cost)
        #pdb.set_trace()
        # plain minima
        fine_depths[l] = np.argmin(sgm_cost, axis=2)
        
        # interpolated minima and values
        fine_depths_interp[l], fine_depths_val[l] = rtxdisp.cost_minima_interp(sgm_cost, disp)

        if i%1000==0:
            print("max interp: {0}".format(np.amax(fine_depths_interp[l])))
        # confidence measure used in "Real-Time Visibility-Based Fusion of Depth Maps"
        # substract 1 at the end since the "real" optimium is included in the vectorized operations
        
       
        #confidence[l][fine_depths_val[l] > min_thresh] = 0.0

        # plain winner takes all minima
        wta_depths[l] = np.argmin(F, axis=2)
        
        # interpolated minima and values from the unregularized cost volume
        wta_depths_interp[l], wta_depths_val[l] = rtxdisp.cost_minima_interp(F, disp)

        confidence[l] = np.sum(np.exp(-((sgm_cost - fine_depths_val[l][:, :, None])**2) / conf_sigma), axis=2) - 1

        #TODO: calculate wta confidence, scale the sigma accordingly
        #confidence[l] = np.sum(np.exp(-((F - wta_depths_val[l][:, :, None])**2) / conf_sigma), axis=2) - 1
        
        # avoid overflow in division
        ind = confidence[l] > eps
        confidence[l][confidence[l] <= 0] = 0.0
        confidence[l][ind] = 1.0 / confidence[l][ind]
        
    return fine_depths, fine_depths_interp, fine_depths_val, wta_depths, wta_depths_interp, wta_depths_val, confidence
    
def calc_costs_selective_with_lut(lenses, disparities, nb_strategy, technique, nb_args, max_cost, refine=True, progress_hook=print):
    
    """
    it firstly calculates the fine and coarse depth map based on the first "circle" (HEX_OFFSETS[1]) with lenses of same focal lens
    then it adds the other lenses (based on strategy, but the first one is always the same) and either 
    - refine the values or
    - substitute the values
    
    Then it merges the costs and returns fine and coarse
    """
    coarse_costs = dict()
    coarse_costs_merged = dict()
    fine_costs = dict()
    lens_std = dict()
    num_lenses = len(lenses)
    num_targets = 0
    
    # 4+2
    # using four lenses from the first circle (the 4 corners of a virtual rectangle around the lens)
    # + 2 lenses that are the closest one to the center lens
    pos1 = [[-1,-1],[-1,2],[1,1],[1,-2],[0,-1],[0,1]]
    pos2 = [[-1,-1],[-2,1],[1,1],[2,-1],[0,-1],[0,1]]
    pos3 = [[-2,1],[-1,2],[2,-1],[1,-2],[0,-1],[0,1]]

    for i, lcoord in enumerate(lenses):
        
        lens = lenses[lcoord]
     
        # some lenses have troubles, mainly the 0-type lenses, when they are far away
        # using this solution seems better   
        if rtxhexgrid.hex_focal_type(lcoord)==0:
            pos = rtxhexgrid.HEX_OFFSETS[1]
        elif rtxhexgrid.hex_focal_type(lcoord)==1:
            pos = pos1
        elif rtxhexgrid.hex_focal_type(lcoord)==2:
            pos = pos1
        else:
            pdb.set_trace()
            
        nb_lenses = _rel_to_abs(lcoord, lenses, pos)
       
        
        if i % 100 == 0:
            progress_hook("Processing lens {0}/{1} - {2}".format(i, num_lenses, lcoord))
            
        # calculate a first guess of the disparity based on the first circle
        fine, coarse, coarse_merged, lens_var = calc_costs_per_lens(lens, nb_lenses, disparities, max_cost, technique)
        nb_offsets, curr_disp_avg = nb_strategy(lens, lenses, coarse, disparities, max_cost=max_cost, nb_args=nb_args)
        
        nb_lenses = _rel_to_abs(lcoord, lenses, nb_offsets)

        if len(nb_lenses) > 0:
            
            #progress_hook("Refined nb: {0}".format(nb_offsets))
            fine_2, coarse_2, _, _ = calc_costs_per_lens(lens, nb_lenses, disparities, max_cost, technique)

            if refine is True:
                fine = np.append(fine, fine_2, axis=0)
                coarse = np.append(coarse, coarse_2, axis=0)
            else:
                fine = fine_2
                coarse = coarse_2

        num_targets += len(fine)
               
        coarse_costs_merged[lcoord]  =  rtxdisp.merge_costs_additive(coarse, max_cost)
        coarse_costs[lcoord] = coarse
        fine_costs[lcoord] = np.array(rtxdisp.merge_costs_additive(fine, max_cost))
    
        lens_std[lcoord] = lens_var

    progress_hook("Num comparisons: {0}".format(num_targets))
    
    return fine_costs, coarse_costs, coarse_costs_merged, lens_std, num_targets, 0.0   
   
def calc_costs_per_lens(lens, nb_lenses, disparities, max_cost, technique):

    cost, img, d = rtxdisp.lens_sweep(lens, nb_lenses, disparities, technique, max_cost=max_cost)
    #print("cost shape: {0}".format(cost.shape))
    coarse_costs = rtxdisp.sweep_to_shift_costs(cost, max_cost)
    #print("Cost shape: {0}".format(coarse_costs.shape))
    #print("Coarse costs: {0}".format(coarse_costs))
    #print("Max cost: {0}".format(max_cost))
    coarse_costs_merged = rtxdisp.merge_costs_additive(coarse_costs, max_cost)
    lens_std = np.std(lens.img[lens.mask > 0])
  
    return cost, coarse_costs, coarse_costs_merged, lens_std
    
class EvalParameters(object):

    def __init__(self):

        self.max_disp_fac = 0.3
        self.min_disp_fac = 0.02 
        self.max_ring = 7
        self.max_cost = 10.0
        self.penalty1 = 0.01 
        self.penalty2 = 0.03 
        self.method = 'plain'
        self.use_rings = '0,1'
        self.refine = True
        self.coc_thresh = 1.2#1.5
        self.conf_sigma = 0.2
        self.max_conf = 2.0
        self.filename = None
        self.coarse = False
        self.coarse_weight = 0.01
        self.struct_var = 0.01
        self.coarse_penalty1 = 0.01
        self.coarse_penalty2 = 0.03
        self.technique = 'sad'
        self.lut_trade_off = 1
        self.num_disp = 12
        self.method = 'real_lut'
