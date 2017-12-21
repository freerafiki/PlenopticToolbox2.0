"""
It contains the information about camera parameters for transforming between world and local coordinates
----
@veresion v1 - December 2017
@author Luca Palmieri
"""

import numpy as np

class Camera(object):
    '''
    World coordinate system:
    x: left->right: negative -> positive
    y: up->down: negative -> positive
    z: looks along the positive z-axis

    Image coordinate system:
    Upper left corner: (0, 0)
    
    '''
    def __init__(self, img=None, focal_length=1, position=None,
                 rotation=None, pixel_size=1, num_channels=1):

        if position is None:
            self.position = np.array([0, 0, 0])
        else:
            self.position = position
            
            
        if rotation is None:
            self.rotation = np.eye(3)
        else:
            self.rotation = rotation

    
        self.focal_length = focal_length
        self.pixel_size = pixel_size
        self.focal_length_px = focal_length / pixel_size
        
        if img is None:
            side=1024
            self.sensor_width = side#256
            self.sensor_height = side#256
            self.num_channels = num_channels
            self.img = np.zeros((self.sensor_width, self.sensor_height, self.num_channels))
        else:
            if len(img.shape) > 2:                
                self.sensor_height, self.sensor_width, self.num_channels = img.shape
            else:
                self.sensor_height, self.sensor_width = img.shape
                self.num_channels = 1
                
            self.img = np.copy(img)

        # the principal point
        self.px = (self.sensor_width - 1.) / 2
        self.py = (self.sensor_height - 1.) / 2

        self.skew = 1

        self._gen_P()
        
    def _gen_K(self):
        
        K = np.array([[self.focal_length_px, 0, self.px],
                      [0, self.focal_length_px, self.py],
                      [0, 0, 1]])

        self.K = K
        self.invK = np.linalg.inv(K)

    def _gen_RT(self):

        R = self.rotation
        t = self.position
        RT = np.zeros((4, 4))


        RT[:3, :3] = R
        RT[:3, 3] = t
        RT[3, 3] = 1.0

        self.RT = RT

    def _gen_P(self):

        self._gen_K()
        self._gen_RT()

       # TODO: switch x, y
        P = np.zeros((3, 4))
        P[:3, :3] = self.rotation.T
        P[:3, 3] = -(np.dot(self.rotation.T, self.position))

        P = np.dot(self.K, P)
        #print("P: {0}, pos: {1}, rot: {2}".format(P, self.position, self.rotation))
        self.P = P

    def project_points(self, p3d):

        p3d = np.asarray(p3d)
        
        p3d = np.hstack((p3d, np.ones((p3d.shape[0], 1))))
        p3d = np.dot(self.P, p3d.T).T
        p3d /= p3d[:, 2][:, None]

        return np.hstack((p3d[:, 1], p3d[:, 0]))

    
    def reproject_img(self, depth_img):

        assert depth_img.shape[:2] == self.img.shape[:2]

    def reproject_points(self, p2d, depths):

        p2d = np.asarray(p2d)
        depths = np.asarray(depths)
        
        assert len(p2d) == len(depths)

        p3d = []
        
        for i, p in enumerate(p2d):
            d = depths[i]

            q = self.pixel_pos_local(p[0], p[1])
            q /= np.linalg.norm(q)
            q *= d
            p3d.append(q)

        p3d = self.transform_world(p3d)
        
        return np.array(p3d)

    def transform_world(self, p3d):
        
        p3d = np.asarray(p3d)
        p3d = np.dot(self.rotation, p3d.T).T
        p3d += self.position

        return p3d
    
    def pixel_pos_local(self, i, j):

        p = np.dot(self.invK, np.array([j, i, 1.]))

        return p 

    def pixel_pos_world(self, i, j):

        p = self.pixel_pos_local(i, j)
        p *= self.focal_length
        p = np.dot(self.rotation, p)
        p += self.position

        return p
        
        
