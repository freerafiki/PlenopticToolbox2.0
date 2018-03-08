# PythonLibrary v1

This folder contains the python library to work with Plenoptic Images. 

It contains the following folders:
- plenopticIO: Input and Output management for such images
- microlens: The object structure used for each micro-lens is built here
- disparity: It contains the code for estimating disparity map from a single image
- rendering: It takes care of the rendering process, showing the image and calculating the all-in-focus images
- sample: Some samples to show how the code works (check the sample folder for a more detailed explanation)

And the following files:
- setup.py: used for the cython compilation of the sgm file
- plenopticGUI.py: a basic graphical user interface to load images and estimate disparity. It is still a beta version for testing, it probably has some bugs

#### Dependencies and requirements
In order to work with the python library, you need:
- version of python 3 or higher (using python3.6)
- a C++ compiler and Cython (version 3 or higher is preferable)
- matplot libraries (python3-matplotlib)
- tk for the GUI (python3-tk)
- numpy package (python3-numpy)
- scipy package (python3-scipy)
- header files (python3-dev)
- opencv for python (python3-opencv)

##### Cython

Once you have installed dependencies, you have to run the setup file `setup.py` that will _cythonize_ the Cython files.
Use the command
```
python setup.py build_ext --inplace
```
It should create one or two files (.so) in the python folder, and they have to be moved to the disparity folder (still have to figure out a way to do it automatic, sorry)

More detail about the cython compilation [here](http://cython.readthedocs.io/en/latest/src/reference/compilation.html#compiling-with-distutils)

Depending on the OS (it was developed on Linux and tested on Linux and Mac), there may be some small issues due to compatibility of some libraries. If you encounter any problem, please send some feedback, it would be useful for correcting.

###### Known issues:
fatal error: 'numpy/arrayobject.h' file not found - Mac OS X 10.10.5 Yosemite, python3.6
workaround found it [here](https://github.com/andersbll/cudarray/issues/52) worked in this case, other possible solutions [here](https://github.com/andersbll/cudarray/issues/25) and [here](https://github.com/hmmlearn/hmmlearn/issues/43)

##### PYTHONPATH

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

##### TEST

To prove that the environmental variable is set, a simple test can be done using the command `printenv` from terminal, that will print the content of all environmental variables.

For further information, write to lpa@informatik.uni-kiel.de
