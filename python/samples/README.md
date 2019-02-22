## Python Samples

The samples given here are intended to give some hints about how to use the available code:

Up to now four scripts are provided, more can be added in case of request (just write me at lpa@informatik.uni-kiel.de)

### Read the image (read_image.py)

Super simple file to test that everything is working. It just reads a file (real or synthetic) by giving as input the path to the configuration file (the .xml, (name of image and configuration file should be the same!) or the .json) and shows the image.

### Change the format of the synthetic images (from_synth_to_xml.py)

It reads all micro-images contained in the given folder (where the scene.json file is) and save the image as .png file with his relative .xml configuration file, so that it looks like a real image and can be easier accessed.

### Estimate the disparity (disparity_sample.py)

Input Image                |  Disparity Map
:-------------------------:|:-------------------------:
![Input Image](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/blob/master/THUMBNAILS/Dragon_Processed%20copy.jpg)  |  ![](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/blob/master/THUMBNAILS/Dragon_Disparity.jpg)

A short example of how to read an image and calculate its disparity map. Several parameters can be tuned, starting from different similarity measures (SAD, SSD, CENSUS) and other algorithm parameters (minimum and maximum disparity, disparity step, penalty function, threshold for circle of confusion, minimum coverage).

I added another feature and now it saves a confidence map (you can change the method to calculate confidence from parameters, check the python file for the comments). That can be used to cut the wrong disparity values and to do some post-processing on the disparity.

For a further insight about the disparity estimation process, please refer to the papers
- [_Palmieri L. , Op Het Veld R. and Koch R, Plenoptic Toolbox 2.0: Benchmarking of Depth Estimation Methods for MLA-based Focused Plenoptic Cameras_](http://data.mip.informatik.uni-kiel.de:555/wwwadmin/Publica/2018/2018_Palmieri_The%20Plenoptic%202.0%20Toolbox:%20Benchmarking%20of%20Depth%20Estimation%20Methods%20for%20MLA-Based%20Focused%20Plenoptic%20Cameras.pdf)
- [_Fleischmann O. and Koch R, Lens-Based Depth Estimation for Multi-focus Plenoptic Cameras, GCPR, 2014_](https://link.springer.com/content/pdf/10.1007/978-3-319-11752-2_33.pdf)
- [_Palmieri L. and Koch R, Optimizing the Lens Selection Process for Multi-Focus Plenoptic Cameras and Numerical Evaluation, LF4CV @ CVPR, 2017_](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w27/papers/Palmieri_Optimizing_the_Lens_CVPR_2017_paper.pdf)

Example run:

`python3 disparity_sample.py ~/path_to_file.xml -dmin 0 -dmax 10 -err False -scene 'synth'`

where `~/path_to_file.xml` is the path to the .xml file **(NB: the script now assumes that image and configuration file have same name, just different extensions, so image should be `~/path_to_file.png`)**, `-dmin` and `-dmax` are respectively the minimum and maximum disparity, `-err True` enables the error analysis (it will work only if a ground truth is available, so only for synthetic images and `-scene 'synth'` describe scene type (`'synth'` = synthetic or `'real'` = real).


### Create a refocused image (refocused_img.py)

There are different ways of obtaining high-quality refocused images. One is for example selecting a patch of pixel from each micro-image and tiling them together (as explained in 2010 Georgiev paper _Reducing Plenoptic Artifacts_)
A simple version of this idea is here implemented to otbain a so-called all-in-focus (or TotalFocus if you are familiar with Raytrix software) image.

Input Image                |  Refocused Image          | Input Image               |  Refocused Image
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![Input Image](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/blob/master/THUMBNAILS/Dragon_Processed%20copy.jpg)  |  ![](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/blob/master/THUMBNAILS/dragon76_ref_img.png) |  ![](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/blob/master/THUMBNAILS/Plant_small.png) |  ![](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/blob/master/THUMBNAILS/Plant76_ref_img.png)

Example run:

`python3 refocused_img.py ~/img.png -disp ~/path_disp_img.png -cfg ~/config.xml -o ~/output_folder/ -name outputName -plus True`

Where `-disp` is the path to the disparity image, `-cfg` to the .xml file, `-name` the preferred output name and `-plus` enable saving the quantization map for debug

### Create perspective views (perspective_views.py)

Using the same idea of the refocused image, we can shift the point where we extract the patches from and create a series of perspective views to simulate a grid of images. These images are created without taking into account the three lens type! To take them into account, check next sample script below!

Sample GIF               |  Sample GIF          |
:-------------------------:|:-------------------------:|
![](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/blob/master/ANIMATIONS/small_ani_dragon.gif)  |  ![](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/blob/master/ANIMATIONS/small_ani_cards.gif) |  

This is a new version and it should be run using the .json file produced from disparity_sample.json. The .json file contains all parameters (and filename of the images produced) so it makes it easier to run several scripts. This way there are no parameters that have to be tuned.

Example run:

`python3 perspective_views.py ~/path_to_the___parameters.json -hv N -vv M`

Where `-hv` and `-vv` are the number of viewpoints in x and y direction, so that you will get `NxM` views.
The script will create a folder named `Views` and three subfolder named `Color, Disps, Other` where the views will be saved.

### Create perspective views using only focused microlenses (focused_perspective_views.py)

Using the same idea described above, but this time creating an image for each lens type and combining them using the disparity information (and the range of each microlens as explained in the .xml generated from RxLive software at the calibration) we can obtain one image. This is smaller, since we use less information (1/3 of micro-images), but should be more accurate.

Sample GIF               |  Sample GIF          |
:-------------------------:|:-------------------------:|
![](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/blob/master/ANIMATIONS/small_ani_balls.gif)  |  ![](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/blob/master/ANIMATIONS/small_ani_plantball.gif) |  

This is a new version and it should be run using the .json file produced from disparity_sample.json. The .json file contains all parameters (and filename of the images produced) so it makes it easier to run several scripts. This way there are no parameters that have to be tuned.

Example run:

`python3 focused_perspective_views.py ~/path_to_the___parameters.json -hv N -vv M`

Where `-hv` and `-vv` are the number of viewpoints in x and y direction, so that you will get `NxM` views.
The script will create a folder named `Views` and three subfolder named `Color, Disps, Other` where the views will be saved.
