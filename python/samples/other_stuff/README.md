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


