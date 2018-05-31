Title: PlenopticToolbox2.0  
Author: Luca Palmieri  
Address: lpa@informatik.uni-kiel.de  
Date:	Februar 2018

# Plenoptic Toolbox 2.0
The Plenoptic Toolbox 2.0 aims to help promote research using Focused Plenoptic Cameras (a.k.a. Plenoptic 2.0), 
providing a dataset of real and synthetic images.

# Related Publications
- [Luca Palmieri, Ron Op Het Veld and Reinhard Koch, _The Plenoptic 2.0 Toolbox: Benchmarking of depth estimation methods for MLA-based focused plenoptic cameras_, ICIP, 2018. (accepted, to appear - preprint version)()
- [_Fleischmann O. and Koch R, Lens-Based Depth Estimation for Multi-focus Plenoptic Cameras, GCPR, 2014_](https://link.springer.com/content/pdf/10.1007/978-3-319-11752-2_33.pdf)
- [_Palmieri L. and Koch R, Optimizing the Lens Selection Process for Multi-Focus Plenoptic Cameras and Numerical Evaluation, LF4CV @ CVPR, 2017_](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w27/papers/Palmieri_Optimizing_the_Lens_CVPR_2017_paper.pdf)



## Code
Inside the python folder there are all information regarding the code structure and how to use it, with some sample that can be used as a reference. Code is developed in Python with some parts connected to C via Cython.
The code is completely open source and can be integrated and further developed for research projects.

## Images
It also provide a python library to work with such images (in particular with Raytrix images at the moment).

### Real Images

Plant Image                |  Plant Depth              | Dragon Image              | Dragon Depth              |
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
![Plant](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/blob/master/THUMBNAILS/Plant_small.png) | ![Plant](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/blob/master/THUMBNAILS/Plant_DEPTH_small.png) | ![Dragon](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/blob/master/THUMBNAILS/Dragon_small.png) | ![Dragon](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/blob/master/THUMBNAILS/Dragon_DEPTH_small.png)


### Synthetic Images

Alley                      |  Alley Depth              | Coffee Image              | Coffee Depth              |
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
![Plant](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/blob/master/THUMBNAILS/alley_light.png) | ![Plant](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/blob/master/THUMBNAILS/alley_light_disp.png) | ![Dragon](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/blob/master/THUMBNAILS/coffee_rose_largest_small.png) | ![Dragon](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/blob/master/THUMBNAILS/coffee_rose_largest_disp_small.png)

The Dataset containing the images (for reasons of space cannot be provided here) is available [here](https://drive.google.com/open?id=17I6nTf4GLYiO9fdWITEy155F-OaonaeQ) 

The Google Drive folder contains real images taken with Raytrix Cameras (R29 and R42) and Synthetic Images (generated with Blender) along with the corresponding calibration or configuration file needed to work on the images.





