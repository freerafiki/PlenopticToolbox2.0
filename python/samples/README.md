## Python Samples

The samples given here are intended to give some hints about how to use the available code:

Up to now four scripts are provided, more can be added in case of request (just write me at lpa@informatik.uni-kiel.de)

#### Read the image and change between micro-images and subaperture images (subapertureimages.py).

![Input Image](Dragon_Processed copy.jpg) ![Subaperture Views](Dragon_Processed_subapertureviews copy.jpg)

Since many people are used to the Lytro (Plenoptic 1.0 Cameras) this script allows you to exchange the way the data is shown. Plenoptic 2.0 Cameras have different physical properties, thus the subaperture views are quite small and low-quality. The script is mostly thought as an example of what you would get treating these kind of images as if they were Lytro images.



#### Crop the image (crop_image.py) (NOT READY YET)

Since the images are taken with high-resolution cameras and such type of images require a high computational effort, the calculations may last several minutes and make it quite annoying, therefore I provide a script to crop an image in order to speed up calculations for testing. The code is for research and prototyping purposes, using Python and C (via Cython) to guarantee a trade-off between usability (debugging and testing) and performance, but is not optimized for real-time and on less powerful computer can be slow.

#### Estimate the disparity (disparity_sample.py)

A short example of how to read an image and calculate its disparity map. Several parameters can be tuned, starting from different similarity measures (SAD, SSD, CENSUS) and other algorithm parameters (minimum and maximum disparity, disparity step, penalty function, threshold for circle of confusion, minimum coverage).

For a further insight about the disparity estimation process, please refer to the papers
- [_Fleischmann O. and Koch R, Lens-Based Depth Estimation for Multi-focus Plenoptic Cameras, GCPR, 2014_](https://link.springer.com/content/pdf/10.1007/978-3-319-11752-2_33.pdf)
- [_Palmieri L. and Koch R, Optimizing the Lens Selection Process for Multi-Focus Plenoptic Cameras and Numerical Evaluation, LF4CV @ CVPR, 2017_](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w27/papers/Palmieri_Optimizing_the_Lens_CVPR_2017_paper.pdf)

#### Create the all-in-focus image (allinfocus.py) (NOT READY YET)

Using both the colored image and the corresponding disparity map as input, it creates both all-in-focus images. It can be useful for visualization or comparison purposes, but it is at the actual point quite basic and not optimal.
