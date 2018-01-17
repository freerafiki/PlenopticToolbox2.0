# PythonLibrary v1

This folder contains the python library to work with Plenoptic Images. 

It contains the following folders:
- plenopticIO: Input and Output management for such images
- microlens: The object structure used for each micro-lens is built here
- disparity: It contains the code for estimating disparity map from a single image
- rendering: It takes care of the rendering process, showing the image and calculating the all-in-focus images
- sample: Some samples to show how the code works (check the sample folder for a more detailed explanation)

#### Dependencies and requirements
In order to work with the python library, you need:
- version of python 3 or higher installed in your computer.
- a C++ compiler

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

To prove that the environmental variable is set, a simple test can be done using the command `printenv` from terminal, that will print the content of all environmental variables.

For further information, write to lpa@informatik.uni-kiel.de
