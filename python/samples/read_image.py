import plenopticIO.imgIO as imgIO
import rendering.render as rnd
import matplotlib.pyplot as plt
import sys

path = sys.argv[1]

# read the image and store it into a dictionary with all informations
lenses, scene_type = imgIO.load_scene(path)

# create another dictionary with only colored image (no other informations stored)
lens_imgs = dict()

for key in lenses:
    lens_imgs[key] = lenses[key].col_img
    
# render the iamge
img = rnd.render_lens_imgs(lenses, lens_imgs)
    
# show the image
plt.imshow(img)
plt.show()    
