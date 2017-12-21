"""
Lens object, with his components
----
@veresion v1 - December 2017
@author Luca Palmieri
"""

import numpy as np
import scipy.interpolate as sinterp
import camera.camera as rtxcam


class Lens(rtxcam.Camera):

    def __init__(self, lcenter=None, pcenter=None, diameter=None,
                 focal_type=None, img=None, focal_length=1,
                 position=None, rotation=None, pixel_size=1,
                 num_channels=1):

        self.lcoord = np.array(lcenter)
        self.pcoord = np.array(pcenter)
        self.focal_type = focal_type
        self.focus_distance = 0
        self.focus_disparity = 0   

        self.data = dict()
        self.grid = None
     
        self.img_grayscale = None
        self.img_interp = None
        self.img_interp3c = [None, None, None] 
        self.depth_img = None
        self.disp_img = None
        self.diameter = diameter
        
        if self.diameter is not None:
            self.radius = diameter / 2
        else:
            self.radius = None

        self.fstop = diameter
        self.inner_radius = self.radius
        self.mask = None
        
        self.lens_border = 1

        self.focal_length = 0
        self.pixel_size = 1
        self.rotation = np.eye(3)
        self.position = np.array([0, 0, 0])
        self.img  = None
        self.col_img = None
        self.col_img_uint = None

        super(Lens, self).__init__(img=img,
                                   focal_length=focal_length,
                                   position=position,
                                   rotation=rotation,
                                   pixel_size=pixel_size,
                                   num_channels=num_channels)
   
          
class LocalLensGrid(object):

    def __init__(self, diameter):

        # lens diameter in pixels
        self.diameter = diameter

        # lens radius
        self.radius = diameter / 2.0

        num_samples = np.ceil(diameter)
        
        x = np.linspace(-1, 1, num_samples) * self.radius
        xx, yy = np.meshgrid(x, x)
     
        # the local lens grid, 2d
        self.xx, self.yy = xx, yy

        # the local lens grid, 1d axes which span the grid
        # these axes are useful for faster interpolated evaluation
        self.x, self.y = self.xx[0, :], self.yy[:, 0]

        # subsample ratio, useful for later disparity calculations
        self.subsample_ratio = 1.0 * num_samples / diameter
        self.shape = self.xx.shape

    def rotate(self, src_center, dst_center):

        src_center = np.asarray(src_center)
        dst_center = np.asarray(dst_center)
        
        # normalized direction between the centers
        v = dst_center - src_center
        v = v / np.linalg.norm(v)

        # rotation matrix according to the orientation difference
        # vector between the lenses
        R = np.array([[v[1], -v[0]], [v[0], v[1]]]).T

        # rotate the grid
        gridy, gridx = np.dot(R, np.vstack((np.ravel(self.yy), np.ravel(self.xx))))
        
        return gridy.reshape(self.shape), gridx.reshape(self.shape)

