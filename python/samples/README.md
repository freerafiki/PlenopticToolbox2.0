## Python Samples

The samples given here are intended to give some hints about how to use the available code:

Now three major scripts are provided, more can be added in case of request (just write me at lpa@informatik.uni-kiel.de)

Older scripts or basic programs are also available in the [other_stuff](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/tree/master/python/samples/other_stuff) page.

### Estimate the disparity (disparity_sample.py)

Input Image                |  Disparity Map
:-------------------------:|:-------------------------:
![Input Image](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/blob/master/THUMBNAILS/Dragon_Processed%20copy.jpg)  |  ![](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/blob/master/THUMBNAILS/Dragon_Disparity.jpg)

A short example of how to read an image and calculate its disparity map. Several parameters can be tuned, starting from different similarity measures (SAD, SSD, CENSUS) and other algorithm parameters (minimum and maximum disparity, disparity step, penalty function, threshold for circle of confusion, minimum coverage).

I added another feature and now it saves a confidence map (you can change the method to calculate confidence from parameters, check the python file for the comments). That can be used to cut the wrong disparity values and to do some post-processing on the disparity.

For a further insight about the disparity estimation process, please refer to the papers
- [_Palmieri L. , Op Het Veld R. and Koch R, Plenoptic Toolbox 2.0: Benchmarking of Depth Estimation Methods for MLA-based Focused Plenoptic Cameras, ICIP, 2018_](http://data.mip.informatik.uni-kiel.de:555/wwwadmin/Publica/2018/2018_Palmieri_The%20Plenoptic%202.0%20Toolbox:%20Benchmarking%20of%20Depth%20Estimation%20Methods%20for%20MLA-Based%20Focused%20Plenoptic%20Cameras.pdf)
- [_Fleischmann O. and Koch R, Lens-Based Depth Estimation for Multi-focus Plenoptic Cameras, GCPR, 2014_](https://link.springer.com/content/pdf/10.1007/978-3-319-11752-2_33.pdf)
- [_Palmieri L. and Koch R, Optimizing the Lens Selection Process for Multi-Focus Plenoptic Cameras and Numerical Evaluation, LF4CV @ CVPR, 2017_](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w27/papers/Palmieri_Optimizing_the_Lens_CVPR_2017_paper.pdf)

Example run:

`python3 disparity_sample.py ~/path_to_file.xml -dmin 0 -dmax 10 -err False -scene 'real'`

where `~/path_to_file.xml` is the path to the .xml file **(NB: the script now assumes that image and configuration file have same name, just different extensions, so image should be `~/path_to_file.png`)**, `-dmin` and `-dmax` are respectively the minimum and maximum disparity, `-err True` enables the error analysis (it will work only if a ground truth is available, so only for synthetic images) and `-scene 'real'` describe scene type (`'synth'` = synthetic or `'real'` = real).



### Rendering: create high quality perspective views  (render_view_2g.py)

The rendering process is similar to the one described in Georgiev paper (_Reducing Plenoptic Artifacts_), with a little more focus on the extraction of the patches. While the initial method was selecting the pixels to be tiled together to create the rendered image, now I sample pixel colors (at decimal values) and create a patch using this values. So this allows to have a slightly better image (some artifacts appearing with the old method are reduced) and also to control the parameters more, the number of images and the disparity shift between them (since we are not constraint from integer, we can have a wide range). That means we can create an almost arbitrary number of images (a lot of images with very low parallax or fewer images with more parallax). Also we can control the sampling in terms of resolution, that means if we would have to select an area of 5 pixel, we can sample 5 times (so take each pixel) or 10 times (extracting a value every half of a pixel) and so on. So using this sampling value we can control resolution, meaning we can create small (around 600x400, like Lytro, lower I guess it won't be that useful, but is possible) or large (around 1600x1000 or even higher if needed) subaperture images.

Sample GIF               |  Sample GIF          |
:-------------------------:|:-------------------------:|
![](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/blob/master/ANIMATIONS/RTX055_resize60.gif)  |  ![](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/blob/master/ANIMATIONS/RTX055_3x3.gif) |  
Here many images (169) with very low parallax and low resolution | On the right, fewer images (9) with higher parallax and slightly higher resolution |
Parameters: `N=13, M=13, J=0.35, S=7`| Parameters: `N=3, M=3, J=2.0, S=9` |

This is a new version and it should be run using the .json file produced from disparity_sample.json. The .json file contains all parameters (and filename of the images produced) so it makes it easier to run several scripts. This way there are no parameters that have to be tuned.

Example run:

`python3 render_view_2g.py ~/path_to_the___parameters.json -hv N -vv M -j J -spl S`

Where `-hv` and `-vv` are the number of viewpoints in x and y direction, so that you will get `NxM` views. Here `J` is a factor applied to the jump (or shift) between views (as said this can be decimal, for experience when the number of views is more than 7x7, jump should be less than one, but of course, it depends from the optical setup and the distance between object and camera, so take this as a suggestion only). Of course, the shift is calculated from the disparity, but a constant factor is applied. Lastly, `S` is the sampling pattern (for now is same in both direction), a large value will result in high resolution subaperture images (like 13 or 15 will bring something like 1600x1000 pixels images) a small value will create smaller images (7 samples should account for around 600x400).
The script will create a folder named `FocusedViews_NxM` and three subfolder named `Color, Disps, Other` where the views will be saved.
The script saves the images in a .png format, if you want to create an animated view or convert it to a matlab matrix (to get the same as Lytro images decoded using the LightField toolbox, check out the [scripts page](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/tree/master/scripts)

### Refocusing: create a focal stack (render_focal_stack.py)

Using the same rendering process described above while keeping constant the size of the patch to be extracted from each lens, we can obtain a refocusing effect. Using subpixel accuracy allows us to create many different planes in focus. At the moment the code creates an image where only one plane is in focus, a more sophisticated version with different planes in focus could be addressed. 
We constrain the focal planes with the size of the patch. Objects that have less than 3 virtual depth (so less than 3 repetitions horizontally on the microlens images) are the hardest, the refocusing works very good in the range from 3 to higher virtual depth (in the tested images there were no upper bound yet).

Sample GIF               |  Sample GIF          |
:-------------------------:|:-------------------------:|
![](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/blob/master/ANIMATIONS/focalstack_d20_r70.gif)  |  ![](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/blob/master/ANIMATIONS/RTX031_focalstack_d20_r70.gif) |  

Parameters for the above images: `fmin=0.1 FMAX=0.95 FS=0.025 S=11`

Example run:

`python3 render_focal_stack.py ~/path_to_the___parameters.json -fpmin fmin -fpmax FMAX -fstep FS -spl S`

Where `fpmin` and `fpmax` are respectively the minimum and the maximum focal plane. We constrain the focal plane to be a number between 0 and 1 (not included), where the closer to 0 means refocusing closer to the camera and closer to 1 means refocusing further away from the camera. The code will generate a series of images refocused at different plane each `fstep` (so if `-fpmin=0.1` and `-fstep=0.1`, it will generate one image at 0.1, one at 0.2, and so on, until `fpmax`..).
`spl` is, as in the view rendering, the number of samples to be used. A higher number would result in a larger image, a lower number in a lower resolution image.
