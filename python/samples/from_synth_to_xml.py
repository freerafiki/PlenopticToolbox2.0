import sys
import plenopticIO.imgIO as rtxsio
import rendering.render as rtxrnd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    if len(sys.argv) < 2:
        raise Exception('Missing name of the folder - Check file for explanations!')
        
    scenename = 'scene'
    picname = sys.argv[1] # this should be the name of the scene (Alley, Bowling..)
    # substitute "..synthetic_images" with the path where you saved the synthetic scenes
    lensespath = "/data1/palmieri/Dataset/{0}/{1}.json".format(picname, scenename)

    lenses = rtxsio.load_from_json(lensespath)
    #..synthetic_images/
    real_path = "/data1/palmieri/2018/October/testplenoptic/{0}".format(picname)

    # save the .xml file    
    rtxsio.save_xml(real_path, lenses)
    
    lens_imgs = dict()
    disp_imgs = dict()
    
    for key in lenses:
        lens_imgs[key] = lenses[key].col_img
        disp_imgs[key] = lenses[key].disp_img
    
    img = rtxrnd.render_lens_imgs(lenses, lens_imgs)
    disp = rtxrnd.render_lens_imgs(lenses, disp_imgs)
    
    # save the colored image
    plt.imsave("/data1/palmieri/2018/October/testplenoptic/{0}.png".format(picname), img)
    plt.imsave("/data1/palmieri/2018/October/testplenoptic/{0}_disp.png".format(picname), disp, vmin=np.min(disp), vmax=np.max(disp), cmap='jet')
