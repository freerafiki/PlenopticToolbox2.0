import sys
import plenopticIO.imgIO as rtxsio

if __name__ == '__main__':

    if len(sys.argv) < 2:
        raise Exception('Missing name of the folder')
        
    scenename = 'scene'
    picname = sys.argv[1] # this should be the name of the scene (Alley, Bowling..)
    # substitute "..synthetic_images" with the path where you saved the synthetic scenes
    lensespath = "..synthetic_images/{0}/{1}.json".format(picname, scenename)

    lenses = rtxsio.load(lensespath)

    real_path = "..synthetic_images/{0}".format(picname)

    rtxsio.save_xml(real_path, lenses)
    
