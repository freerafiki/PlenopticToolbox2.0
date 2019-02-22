Here a collection of small scripts that can be useful when using the code!

### GIF

gen_gif is meant to be used to create gif animation for perspective views, check out the [animation page](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/tree/master/ANIMATIONS) for a more complete explanation!



### MATLAB conversion

The script that creates the perspective views creates also a .txt file with the information to convert it to the more popular matlab format.
Using the readLFtxt.m script, you can re-read the images and create a 5-D matlab structure `(N,M,h,w,c)` with `N` and `M` the number of views, `h` and `w` the view resolution and `c` the number of channels (3, RGB).
This way we reach a similar structure of the LF decoded using Dansereau's toolbox from Lytro images for example, for an easier comparison.
