# Python Samples

The samples given here are intended to give some hints about how to use the available code:

Up to now four scripts are provided, more can be added in case of request (just write me at lpa@informatik.uni-kiel.de)

- Read the image and change between micro-images and subaperture images (differentviews.py).

Since many people are used to the Lytro (Plenoptic 1.0 Cameras) this script allows you to exchange the way the data is shown. Plenoptic 2.0 Cameras have different physical properties, thus the subaperture views are quite small and low-quality. The script is mostly thought as an example of what you would get treating these kind of images as if they were Lytro images.

- Crop the image (crop_image.py)

Since the images are taken with high-resolution cameras and such type of images require a high computational effort, the calculations may last several minutes and make it quite annoying, therefore I provide a script to crop an image in order to speed up calculations for testing. The code is for research and prototyping purposes, using Python and C (via Cython) to guarantee a trade-off between usability (debugging and testing) and performance, but is not optimized for real-time and on less powerful computer can be slow.

- Estimate the disparity (disparity_sample.py)

A short example of how to read an image and calculate its disparity map. Several parameters can be tuned, starting from different similarity measures (SAD, SSD, CENSUS) and other algorithm parameters (minimum and maximum disparity, disparity step, penalty function, threshold for circle of confusion, minimum coverage). For more detail about parameters, please refer to the GCPR 2014 Paper "_Oliver Fleischmann and Reinhard Koch. Lens-Based Depth Estimation for Multi-focus Plenoptic Cameras, pages 410–420. Springer International Publishing, 2014_"
