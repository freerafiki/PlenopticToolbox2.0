Here there are more basic examples and older methods. The refocusing image generation was done in a more unefficient way, and there is a version that do not take into account the different lens types.
They are less interesting for the results, but may be useful as a base to try out new things, or as example to play around with the code.


### Read the image (read_image.py)

Super simple file to test that everything is working. It just reads a file (real or synthetic) by giving as input the path to the configuration file (the .xml, (name of image and configuration file should be the same!) or the .json) and shows the image.

### Change the format of the synthetic images (from_synth_to_xml.py)

It reads all micro-images contained in the given folder (where the scene.json file is) and save the image as .png file with his relative .xml configuration file, so that it looks like a real image and can be easier accessed.


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
The script will create a folder named `Views_NxM` and three subfolder named `Color, Disps, Other` where the views will be saved.
The script saves the images in a .png format, if you want to create an animated view or convert it to a matlab matrix (to get the same as Lytro images decoded using the LightField toolbox, check out the [scripts page](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/tree/master/scripts)

### Create perspective views using only focused microlenses (focused_perspective_views.py)

Using the same idea described above, but this time creating an image for each lens type and combining them using the disparity information (and the range of each microlens as explained in the .xml generated from RxLive software at the calibration) we can obtain one image. This is smaller, since we use less information (1/3 of micro-images), but should be more accurate.

Sample GIF               |  Sample GIF          |
:-------------------------:|:-------------------------:|
![](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/blob/master/ANIMATIONS/small_ani_balls.gif)  |  ![](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/blob/master/ANIMATIONS/small_ani_plantball.gif) |  

This is a new version and it should be run using the .json file produced from disparity_sample.json. The .json file contains all parameters (and filename of the images produced) so it makes it easier to run several scripts. This way there are no parameters that have to be tuned.

Example run:

`python3 focused_perspective_views.py ~/path_to_the___parameters.json -hv N -vv M`

Where `-hv` and `-vv` are the number of viewpoints in x and y direction, so that you will get `NxM` views.
The script will create a folder named `FocusedViews_NxM` and three subfolder named `Color, Disps, Other` where the views will be saved.
The script saves the images in a .png format, if you want to create an animated view or convert it to a matlab matrix (to get the same as Lytro images decoded using the LightField toolbox, check out the [scripts page](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/tree/master/scripts)

### Rendering: high quality perspective views (render_view_2g.py)

The rendering process is similar to the one described in Georgiev paper (_Reducing Plenoptic Artifacts_), with a little more focus on the extraction of the patches. While the initial method was selecting the pixels to be tiled together to create the rendered image, now I sample pixel colors (at decimal values) and create a patch using this values. So this allows to have a slightly better image (some artifacts appearing with the old method are reduced) and also to control the parameters more, the number of images and the disparity shift between them (since we are not constraint from integer, we can have a wide range). That means we can create an almost arbitrary number of images (a lot of images with very low parallax or fewer images with more parallax). Also we can control the sampling in terms of resolution, that means if we would have to select an area of 5 pixel, we can sample 5 times (so take each pixel) or 10 times (extracting a value every half of a pixel) and so on. So using this sampling value we can control resolution, meaning we can create small (around 600x400, like Lytro, lower I guess it won't be that useful, but is possible) or large (around 1600x1000 or even higher if needed) subaperture images.

Sample GIF               |  Sample GIF          |
:-------------------------:|:-------------------------:|
![](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/blob/master/ANIMATIONS/RTX055_resize60.gif)  |  ![](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/blob/master/ANIMATIONS/RTX055_3x3.gif) |  
Here many images (169) with very low parallax and low resolution | On the right, fewer images (9) with higher parallax and slightly higher resolution |
Parameters: `N=13, M=13, J=0.35, S=7`| Parameters: `N=3, M=3, J=2.0, S=9` |

This is a new version and it should be run using the .json file produced from disparity_sample.json. The .json file contains all parameters (and filename of the images produced) so it makes it easier to run several scripts. This way there are no parameters that have to be tuned.

Example run:

`python3 render_view_3g.py ~/path_to_the___parameters.json -hv N -vv M -j J -spl S`

Where `-hv` and `-vv` are the number of viewpoints in x and y direction, so that you will get `NxM` views. Here `J` is a factor applied to the jump (or shift) between views (as said this can be decimal, for experience when the number of views is more than 7x7, jump should be less than one, but of course, it depends from the optical setup and the distance between object and camera, so take this as a suggestion only). Of course, the shift is calculated from the disparity, but a constant factor is applied. Lastly, `S` is the sampling pattern (for now is same in both direction), a large value will result in high resolution subaperture images (like 13 or 15 will bring something like 1600x1000 pixels images) a small value will create smaller images (7 samples should account for around 600x400).
The script will create a folder named `FocusedViews_NxM` and three subfolder named `Color, Disps, Other` where the views will be saved.
The script saves the images in a .png format, if you want to create an animated view or convert it to a matlab matrix (to get the same as Lytro images decoded using the LightField toolbox, check out the [scripts page](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/tree/master/scripts)