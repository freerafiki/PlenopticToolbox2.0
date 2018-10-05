from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("disparity/sgm_cost_path.pyx"),
    include_dirs=[numpy.get_include()]
)    

#from distutils.core import setup, Extension
#from Cython.Build import cythonize
#

#setup(
#    ext_modules=[
#        Extension("sgm_cost_path", ["disparity/sgm_cost_path.pyx"],
#                  include_dirs=[numpy.get_include()]),
#    ],
#)
