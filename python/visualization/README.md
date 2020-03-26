## Visualization Scripts

The scripts here are intended for visualization: I used them to create images or 3D views or colored grids to show something.

They do not do special functions or are very important for scientific purposes, yet they can help if you need to prepare something to show to people so they understand.

Since this is usually a annoying task and I did some, I share them.

Samples to do stuff are [here](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/tree/master/python/samples)!

### Create masked image (create_mask.py)

White Mask              |  Lens Type Mask
:-------------------------:|:-------------------------:
![Input Image](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/blob/master/THUMBNAILS/vis/mask.png)  |  ![](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/blob/master/THUMBNAILS/vis/lt_mask.png)
RGB with Mask            |  Disp with Mask
![Input Image](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/blob/master/THUMBNAILS/vis/rgb_m.jpg)  |  ![](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/blob/master/THUMBNAILS/vis/disp_m.jpg)

Here you can see the masks - white and lens types could even be useful - color and disp maybe if you are interested in the behaviour of a particular type of lens. 

Example run:

`python3 disparity_sample.py ~/path_to_file.json --lt --rgb --disp --conf`

where `--lt` enable the saving of the lens type map, `--rgb` of the rgb image, `--disp` of the disparity image and even `--conf` of the confidence (never used).


### Visualization in 3D: just visualize it in 3D (render_and_visualize3D.py/visualize_disp3D.py)

**DISCLAIMER:** This script is not reprojecting the disparity into the depth and creating a pointcloud, is just pulling it and visualizing in 3D (like if you would call the `mesh` function in Matlab). It is intended only as a visualization tool to look at the disparity and to save them as .ply files. 
Ply files can be opened with Open-Source free Softwares like Meshlab or CloudCompare.
Code for the 3D reconstruction to create pointclouds is coming, but not finished - please have patience (or DIY and contribute!)

3D Visualization					|  3D Visualization					|
:-------------------------:|:-------------------------:|
![](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/blob/master/THUMBNAILS/visualize3D_1.png)  |  ![](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/blob/master/THUMBNAILS/visualize3D_2.png) |  

Parameters for the above images: `S=15`

There are two scripts:
 - One follows the format of the other scripts - meaning it accepts as parameter the `.json` file. (`render_and_visualize3D.py`). That means it reads the big image and renders and saves it as disparity and as 3D visualization (as `.ply`) so it may be slower.
 - The second just visualize one already compute disparity, so it takes as parameters the path (relative to the current folder or absolute) of the color image and disp (and a scaling if needed, otherwise it sets automatically to the average between height and width of the image).

Those scripts may be useful if you need to compare two disparity maps that looks similar and you can see the borders in 3D.

Example run: (render_and_visualize3D.py)

`python3 render_and_visualize3D.py  ~/path_to_the___parameters.json -spl S`

Example run: (visualize_disp3D.py)

`python3 visualize_disp3D.py  path_to_the_color_image.png path_to_the_disp.png scaling`


