# PythonLibrary

This folder contains the python library to work with Plenoptic Images.

It contains the following folders:
- plenopticIO: Input and Output management for such images
- microlens: The object structure used for each micro-lens is built here
- disparity: It contains the code for estimating disparity map from a single image
- rendering: It takes care of the rendering process, showing the image and calculating the all-in-focus images
- samples: Some samples to show how the code works (check the [samples folder](https://github.com/freerafiki/PlenopticToolbox2.0/tree/master/python/samples) for a more detailed explanation)

And the following files:
- setup.py: used for the cython compilation of the sgm file
- plenopticGUI.py: a basic graphical user interface to load images and estimate disparity. It is still a beta version for testing, it probably has some bugs

**NOTE:** The GUI was developed long time ago (as of now, end 2024), so it might not be the best option to get all functionalities. 
I suggest you use the examples provided for some inspiration (check out the [samples folder](https://github.com/freerafiki/PlenopticToolbox2.0/tree/master/python/samples)!).
If you feel like you would want a GUI, feel free to reach out and I can try to assist you and I am super open to merge your code for a better GUI!

## Dependencies and requirements
In order to work with the python library, you need:
- version of python 3 or higher (using python3.6)
- a C++ compiler and Cython (version 3 or higher is preferable)
- matplot libraries (python3-matplotlib)
- tk for the GUI (python3-tk)
- numpy package (python3-numpy)
- scipy package (python3-scipy)
- header files (python3-dev)
- opencv for python (python3-opencv)

### Cython

Once you have installed dependencies, you have to run the setup file `setup.py` that will _cythonize_ the Cython files.
Use the command
```
python setup.py build_ext --inplace
```
It should create one or two files (.so) in the python folder, and they have to be moved to the disparity folder (still have to figure out a way to do it automatic, sorry)

More detail about the cython compilation [here](http://cython.readthedocs.io/en/latest/src/reference/compilation.html#compiling-with-distutils)

Depending on the OS (it was developed on Linux and tested on Linux and Mac), there may be some small issues due to compatibility of some libraries. If you encounter any problem, please send some feedback, it would be useful for correcting.

#### Known issues:
fatal error: 'numpy/arrayobject.h' file not found - Mac OS X 10.10.5 Yosemite, python3.6

workaround found it [here](https://github.com/andersbll/cudarray/issues/52) worked in this case, other possible solutions [here](https://github.com/andersbll/cudarray/issues/25) and [here](https://github.com/hmmlearn/hmmlearn/issues/43)

#### PYTHONPATH

Moreover, the structure of the python code requires that you set the environmental variable of your computer
`PYTHONPATH=/thepathwhereyoudownloadedthefolder/PlenopticToolbox/python`

This can be done temporarily in two ways (temporarily means this has to be done everytime the python scripts have to be launched):
- in python, at the beginning of the code, using sys
```python
import sys
sys.path.append('/thepathwhereyoudownloadedthefolder/PlenopticToolbox/python')
```
- in the terminal window, by using the command
```
export PYTHONPATH=/thepathwhereyoudownloadedthefolder/PlenopticToolbox/python
```

There is another way to set the environmental variable at login, by editing the right configuration file, either `~/.bash_profile` or `~/.bashrc` ([here there is a more detailed explanation about which file to edit](https://www.digitalocean.com/community/tutorials/how-to-read-and-set-environmental-and-shell-variables-on-a-linux-vps#setting-environmental-variables-at-login))

In the file it should be added the line
```
export PYTHONPATH=/thepathwhereyoudownloadedthefolder/PlenopticToolbox/python
```

### TEST

To prove that the environmental variable is set, a simple test can be done using the command `printenv` from terminal, that will print the content of all environmental variables.


### RUNNING A SCRIPT

In order to correctly run a script, you need to have downloaded at least one image (not from github as they don't fit here for reason of space, but from the google drive folder or the figshare dataset) and its corresponding configuration file (it ends in .xml)

Once you have both, the script can be run using the corresponding command. For ease of use, the script assumes that the name of the configuration file and the image are the same. The easiest way to do this is just renaming the configuration file (it is the same for many pictures) as the image.
For example, if you want to use _Dragon_Processed.png_ image, change .xml file to _Dragon_Processed.xml_. You are also free to modify the code (in the folder plenopticIo the file [imgIO.py](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/blob/master/python/plenopticIO/imgIO.py) is taking care of this, at the lines 149 and 163 there are two methods that are using the filename, if you edit there putting the name you wish, this can be easily changed).

For more example about scripts, go to the [samples page](https://github.com/PlenopticToolbox/PlenopticToolbox2.0/tree/master/python/samples).

For further information, write to bagigi@disroot.org
