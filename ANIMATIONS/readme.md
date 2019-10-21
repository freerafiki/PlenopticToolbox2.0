# Animation

This page is just as an explanation to create small gifs from images (viewpoints, disparity or so) in order to show what can be done with the images.


Sample GIF               |  Sample GIF          |  Sample GIF          |  Sample GIF          |
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
![](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/blob/master/ANIMATIONS/small_ani_dragon.gif)  |  ![](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/blob/master/ANIMATIONS/small_ani_cards.gif) | ![](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/blob/master/ANIMATIONS/small_ani_glasses.gif) |  ![](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/blob/master/ANIMATIONS/focalstack_d20_r70.gif) | 

#### VIEWS
The script `gen_gif` is a shell script to generate a gif from the views generated from the perspective_views.py script, for example.
It uses the [`convert` command](https://imagemagick.org/script/convert.php) that is part of the [ImageMagick](https://www.imagemagick.org/) package, so you need to install that if you want to use it.

it should be run as 

```
sh gen_gif ~/complete_path_to_the_root_folder_where_you_will_find_the_Views_folder(without_Views_in_the_path)
```

It should output information about what is trying to do.

The script will generate very large gif (using all the images!), so gif can be compressed using `ffmpeg` library:

```
ffmpeg -i animation.gif -vf scale=720:480 small_ani.gif
```

Using of course whatever other resolution you want instead of 720:480 if needed.

#### REFOCUSING
Regarding the focal stack, a gif where the focal plane is moved can be generated using 
```
sh fs_gif '~/complete_path_to_the_root_folder_where_you_will_find_the_Views_folder(without_Views_in_the_path)' time resize spl
```
where `ticks` controls the speed at which the frames are changing (simple version: `FPS = 100/ticks`, so if `ticks=20` we have 5 Frames per second (FPS), more detailed explanation of what ticks are [here](http://www.imagemagick.org/script/command-line-options.php#delay), delay is actually the command used in this case), `resize` applies a resizing factor to the image (so the final resolution will be scaled in percentage (`resize 100` means same resolution, `resize 50` half the resolution) and `spl` is the number of samples you used in the generation of the image (see the [sample page](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/tree/master/python/samples) for detail on that)

PS: It also uses the [`convert` command](https://imagemagick.org/script/convert.php) that is part of the [ImageMagick](https://www.imagemagick.org/) package
