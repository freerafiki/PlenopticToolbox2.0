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
    lensespath = "..synthetic_images/{0}/{1}.json".format(picname, scenename)

    lenses = rtxsio.load_from_json(lensespath)
    #..synthetic_images/
    savingname = picname[-16:]
    outputpath = '/outputpath/'
    real_path = "{0}/{1}".format(outputpath, savingname)

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
    plt.imsave("{0}/{1}.png".format(outputpath, savingname), img)
    disp_output_path = '/outputpath/'
    plt.imsave("{0}/{1}_disp.png".format(disp_output_path, savingname), disp, vmin=np.min(disp), vmax=np.max(disp), cmap='gray')
    plt.imsave("{0}/{1}_disp_col.png".format(disp_output_path, savingname), disp, vmin=np.min(disp), vmax=np.max(disp), cmap='jet')
