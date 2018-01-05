# PythonLibrary v1

This folder contains the python library to work with Plenoptic Images. 

It contains the following folders:
- plenopticIO: Input and Output management for such images
- microlens: The object structure used for each micro-lens is built here
- disparity: It contains the code for estimating disparity map from a single image
- rendering: It takes care of the rendering process, showing the image and calculating the all-in-focus images
- sample: Some samples to show how the code works

In order to work with the python library, you need a version of python 3 or higher installed in your computer.
The structure of the python code requires that you set the environmental variable of your computer 
PYTHONPATH=/thepathwhereyoudownloadedthefolder/PlenopticToolbox/python
This can be done temporarily in two ways:
- in python, at the beginning of the code, using sys
```python
import sys.path
sys.path.append('/thepathwhereyoudownloadedthefolder/PlenopticToolbox/python')
```
- in the terminal window, by using the command
export PYTHONPATH=/thepathwhereyoudownloadedthefolder/PlenopticToolbox/python



For further information, write to lpa@informatik.uni-kiel.de
