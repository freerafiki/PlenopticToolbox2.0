Title: PlenopticToolbox2.0  
Author: Luca Palmieri  
Address: lpa@informatik.uni-kiel.de  
Date:	Februar 2018

# Plenoptic Toolbox 2.0
The Plenoptic Toolbox 2.0 aims to help promote research using Focused Plenoptic Cameras (a.k.a. Plenoptic 2.0), 
providing a dataset of real and synthetic images.

## Code
Inside the python folder there are all information regarding the code structure and how to use it, with some sample that can be used as a reference. Code is developed in Python with some parts connected to C via Cython.
The code is completely open source and can be integrated and further developed for research projects.
The python library allows you to work with such images (in particular with Raytrix images at the moment).

For a more detailed explanation on how to use it, please refer to the [python page](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/tree/master/python).

## Resources
Here there are reference to some dataset we are creating for different applications

### Plenoptic 2.0 Images
#### Real Images

Plant Image                |  Plant Depth              | Dragon Image              | Dragon Depth              |
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
![Plant](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/blob/master/THUMBNAILS/Plant_small.png) | ![Plant](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/blob/master/THUMBNAILS/Plant_DEPTH_small.png) | ![Dragon](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/blob/master/THUMBNAILS/Dragon_small.png) | ![Dragon](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/blob/master/THUMBNAILS/Dragon_DEPTH_small.png)


#### Synthetic Images

Alley                      |  Alley Depth              | Coffee Image              | Coffee Depth              |
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
![Plant](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/blob/master/THUMBNAILS/alley_light.png) | ![Plant](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/blob/master/THUMBNAILS/alley_light_disp.png) | ![Dragon](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/blob/master/THUMBNAILS/coffee_rose_largest_small.png) | ![Dragon](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/blob/master/THUMBNAILS/coffee_rose_largest_disp_small.png)

The Dataset containing the images (for reasons of space cannot be provided here) is available [here](https://drive.google.com/open?id=17I6nTf4GLYiO9fdWITEy155F-OaonaeQ) 

The Google Drive folder contains real images taken with Raytrix Cameras (R29 and R42) and Synthetic Images (generated with Blender) along with the corresponding calibration or configuration file needed to work on the images.

Please if you use the work presented here cite the source
- [Luca Palmieri, Ron Op Het Veld and Reinhard Koch, _The Plenoptic 2.0 Toolbox: Benchmarking of depth estimation methods for MLA-based focused plenoptic cameras_](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/tree/master/Publications/PreprintICIP2018.pdf)


### Matching Plenoptic 1.0 and 2.0 Images - Collaboration with Waqas Ahmad from MidSweden University
Another dataset where same scene is captured with commercial version of 1.0 (Lytro Illum) and 2.0 (Raytrix R29).

[Link to the dataset](https://figshare.com/articles/The_Plenoptic_Dataset/6115487)

- Waqas Ahmad, Luca Palmieri, Reinhard Koch and Mårten Sjöström, _Matching Light Field Datasets from Plenoptic Cameras 1.0 and 2.0_
3DTV, 2018. (accepted, to appear)


### Plenoptic Simulator - Collaboration with Tim Michels and Arne Petersen from Kiel University
Physically based Blender Plug-in to simulate plenoptic 1.0 and 2.0 capturing process.

[Github repository](https://github.com/Arne-Petersen/Plenoptic-Simulation)

- [ Tim Michels, Arne Petersen, Luca Palmieri, Reinhard Koch, _Simulation of Plenoptic Cameras_, 3DTV, 2018. (accepted, to appear - preprint version) ](http://data.mip.informatik.uni-kiel.de:555/wwwadmin/Publica/2018/2018_Michels_Simulation%20of%20Plenoptic%20Cameras.pdf)



