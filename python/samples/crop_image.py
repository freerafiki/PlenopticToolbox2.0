"""
Sample code to read an image and estimate the disparity
Parameters can be input by hand or predefinite ones will be used
----
@veresion v1 - December 2017
@author Luca Palmieri
"""
import argparse
import os
import sys
import json
import pdb
import matplotlib.pyplot as plt
import rendering.render as rtxrender

name_file = sys.argv[1]

lenses = xmlio.load("/home/palmieri/Pictures/R42/Micro/sc2_det2.png", "/home/palmieri/Pictures/R42/Micro/sc2_det2.xml")
#name = '/media/palmieri/VERBATIM/zombie/ZombieD_0000001914_0000000461_0000000020_Processed'
ext = '.bmp'
exto = '.png'
#fullname = "{0}{1}".format(name, ext)
#lenses = xmlio.load(fullname, "/data1/palmieri/Images/Sergio/zombie.xml")


#pdb.set_trace()
lens_imgs = dict()

for key in lenses:
    lens_imgs[key] = lenses[key].col_img
    #plt.imshow(lens_imgs[key])
    #plt.show()
    
    #pdb.set_trace()
pdb.set_trace()

#img, dimensions, shape = dv.map_from_micro_images_to_views(lenses, lens_imgs) 
#pdb.set_trace()
cropped_img = rendering.render_cropped_img(lenses, lens_imgs, 1000, 1000, 500, 500)

img = rendering.render_lens_imgs(lenses, lens_imgs)
#img2 = rendering.render_disp_imgs(lenses, disp_imgs)
plt.imshow(img)
plt.show()
pdb.set_trace()
#pdb.set_trace()
img1, img2, img3 = rendering.render_three_imgs(lenses, lens_imgs)

#plt.subplot(131)
#plt.imshow(img1)
#plt.subplot(132)
#plt.imshow(img2)
#plt.subplot(133)
#plt.imshow(img3)
#plt.show()

im1name = "{0}_1{1}".format(name, exto)
im2name = "{0}_2{1}".format(name, exto)
im3name = "{0}_3{1}".format(name, exto)
plt.imsave(im1name, img1)
plt.imsave(im2name, img2)
plt.imsave(im3name, img3)

pdb.set_trace()

#plt.figure(2)
#plt.imshow(img2[0])

#plt.figure(2)

#plt.imshow(img[1])

plt.show()
pdb.set_trace()
#[img, patched_img]

plt.imshow(img[1])
plt.show()
plt.imsave("/home/palmieri/Pictures/DATASET/Colors3/color_2.png", img[2]) #, vmin=0, vmax=8.0)
#plt.imsave("/home/palmieri/Pictures/DATASET/Colors3/color_all.png", patched_img) #, vmin=0, vmax=8.0)
