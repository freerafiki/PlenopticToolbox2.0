"""
File taking care of managing the images, reading and saving
It works with both real (with .xml calibration file) and synthetic images (with scene.json file)
The structure used to store micro-lens images is a python dictionary with lens element
----
@veresion v1.1 - Januar 2017
@author Luca Palmieri
"""

import numpy as np
import scipy.interpolate as sinterp
import matplotlib.pyplot as plt
import plenopticIO.lens_grid as rtxhexgrid
import rendering.render as rtxrnd
import microlens.lens as rtxlens
from xml.etree import ElementTree
import pdb
import os
import json
import string

def _float_from_element(parent, element):

    return float(parent.find(element).text)
         
def _floats_from_section(parent, section, elements):

    sec = parent.findall(section)
    
    vals = dict()

    if len(sec) != 1:
        raise ValueError("Number of {0} entries != 1".format(section))

    sec = sec[0]
 
    for e in elements:
        res = sec.findall(e)
        if len(res) != 1:
            raise ValueError("Number of {0} entries != 1".format(e))
        vals[e] = float(res[0].text)

    return vals

def _section_with_attrib(parent, section, attr, val):
    
    elements = parent.findall(section)

    result = None

    for e in elements:
        if e.attrib[attr] == val:
            result = e
            break

    return result

def read_calibration(filename):

    tree = ElementTree.parse(filename)
    root = tree.getroot()

    calibration = dict()

    calibration["offset"] = _floats_from_section(root, "offset", ["x", "y"])
    calibration["diameter"] = _float_from_element(root, "diameter")
    calibration["rotation"] = _float_from_element(root, "rotation")
    calibration["lens_border"] = _float_from_element(root, "lens_border")
    calibration["lens_base_x"] = _floats_from_section(root, "lens_base_x", ["x", "y"])
    calibration["lens_base_y"] = _floats_from_section(root, "lens_base_y", ["x", "y"])
    calibration["sub_grid_base"] = _floats_from_section(root, "sub_grid_base", ["x", "y"])
    
    lens_types = []
    i = 0
    sec = _section_with_attrib(root, "lens_type", "id", str(i))
    
    while sec is not None:
        sub_dict = dict()
        sub_dict["offset"] = _floats_from_section(sec, "offset", ["x", "y"])
        sub_dict["depth_range"] = _floats_from_section(sec, "depth_range", ["min", "max"])
        lens_types.append(sub_dict)
        i += 1
        sec = _section_with_attrib(root, "lens_type", "id", str(i))

    calibration["lens_types"] = lens_types

    return MLACalibration(calibration)

class MLACalibration(object):

    def __init__(self, calib):
        
        # the diameter of a single lens in pixels
        self.lens_diameter = calib['diameter']

        # lens radius in pixels
        self.lens_radius = self.lens_diameter / 2.0

        # pixel coordinates of the center lens
        # raw image origin (0, 0) is upper left corner
        # lower right corner is (h, w)
        x, y = calib['offset']['x'], calib['offset']['y']
        offset = (-y, x)
        self.offset = np.array(offset)

        # rotation angle of the grid, counter-clockwise
        self.rot_angle = calib['rotation']

        # lens border in pixels
        self.lens_border = calib['lens_border']
  
        # lens bases in lens units
        x, y = calib['lens_base_x']['x'], calib['lens_base_x']['y']

        v = (x, y)
        # reflect
        lens_base_x = np.array([-v[1], v[0]])
            
        # lens bases in lens units
        x, y = calib['lens_base_y']['x'], calib['lens_base_y']['y']
        v = (x, y)
        lens_base_y = np.array([-v[1], v[0]])

        lbx = lens_base_x
        lby = -lens_base_y + lens_base_x

        # axial grid basis vectors in lens units
        self.lbasis = np.vstack((lby, lbx)).T

        # axial grid basis vectors in pixels
        self.pbasis = self.lbasis * self.lens_diameter

        self.inner_lens_radius = self.lens_radius - self.lens_border

        # information about the focus range for eacht lens type
        self.lens_types = calib['lens_types']


def _hex_focal_type(c):
    
    """
    Calculates the focal type for the three lens hexagonal grid
    """

    focal_type = ((-c[0] % 3) + c[1]) % 3

    return focal_type 

"""
Decide which method should be used for loading the image
----
October 2018
"""
def load_scene(filename, calc_err=False):

    basename, suffix = os.path.splitext(filename)
    
    if calc_err:
        disp_name = basename + '_disp.png'
        img_filename = basename + '.png'
        lenses = load_with_disp(img_filename, disp_name, filename)
    else:
        if suffix == '.json':
            lenses = load_from_json(filename)
        elif suffix == '.xml':
            img_filename = basename + '.png'
            lenses = load_from_xml(img_filename, filename)

    return lenses
   
   
"""
Load an image and its disparity map
Useful to estimate the error then
-----
October 2018
"""   
def load_with_disp(img_filename, disp_name, config_filename):   

    img = plt.imread(img_filename)
    disp = plt.imread(disp_name)
    calib = read_calibration(config_filename)
    
    if len(img.shape) < 3:
        print("This version works only for colored image")
        raise ValueError("Unsopported image dimension {0}".format(img.shape))    
    elif len(img.shape) == 3:
        weights = np.array([0.3, 0.59, 0.11])
        data = np.sum([img[:, :, i] * weights[i] for i in range(3)], axis=0)
        data_col = img
    else:
        raise ValueError("Unsupported image dimensions {0}".format(img.shape))
        
    # the image grid in pixels used for the bivariate spline interpolation
    gridy, gridx = range(data.shape[0]), range(data.shape[1])

    data_interp = sinterp.RectBivariateSpline(gridy, gridx, data)
    data_interp_r = sinterp.RectBivariateSpline(gridy, gridx, data_col[:,:,0])
    data_interp_g = sinterp.RectBivariateSpline(gridy, gridx, data_col[:,:,1])
    data_interp_b = sinterp.RectBivariateSpline(gridy, gridx, data_col[:,:,2])
    # we need to be using the gray version with the real values
    if disp[:,:,0].all() == disp[:,:,1].all() == disp[:,:,2].all():
        disp_data = disp[:,:,0]
    else:
	    print("Wrong disparity map. Probably trying to read the colored version of the disparity map. Please use the gray version")
	    raise ValueError("Wrong disparity map")    
    disp_interp = sinterp.RectBivariateSpline(gridy, gridx, disp_data)
    img_shape = np.asarray(data.shape[0:2])
    coords = rtxhexgrid.hex_lens_grid(img_shape, calib.lens_diameter, calib.rot_angle, calib.offset, calib.lbasis)
    focal_type_offsets = [tuple(lens['offset'].values()) for lens in calib.lens_types]
    focal_type_offsets = [(int(p[0]), int(p[1])) for p in focal_type_offsets]
    focal_type_map = {_hex_focal_type(p) : i for i, p in enumerate(focal_type_offsets)}

    lenses = dict()
    local_grid = rtxlens.LocalLensGrid(calib.lens_diameter)
    x, y = local_grid.x, local_grid.y
    xx, yy = local_grid.xx, local_grid.yy
    mask = np.zeros_like(local_grid.xx)
    mask[xx**2 + yy**2 < calib.inner_lens_radius**2] = 1    

    for lc in coords:

        pc = coords[lc]
        lens = rtxlens.Lens(lcenter=lc, pcenter=pc, diameter=calib.lens_diameter)
        lens.focal_type = _hex_focal_type(lc)
        lens.fstop = lens.diameter
        lens.pixel_size = 1.0
        lens.position = np.array([pc[1], pc[0], 0])
        cen_x = round(pc[0])
        cen_y = round(pc[1])
        x1 = int(cen_x + round(np.min(x)))
        x2 = int(cen_x + round(np.max(x)))
        y1 = int(cen_y + round(np.min(y)))
        y2 = int(cen_y + round(np.max(y)))
        lens.num_channels=3
        lens.img = data_interp(y + pc[0], x + pc[1])
        lens.disp_img = disp_interp(y + pc[0], x + pc[1])
        lens.img_interp = sinterp.RectBivariateSpline(y, x, lens.img)
        # colored version of the interpolated image:
        lens.img_interp3c[0] = data_interp_r
        lens.img_interp3c[1] = data_interp_g
        lens.img_interp3c[2] = data_interp_b
        lens.col_img_uint = data_col[x1:x2+1, y1:y2+1] #np.zeros((lens.img.shape[0], lens.img.shape[1], lens.num_channels))

        if lens.col_img_uint.dtype == 'uint8':
            new_col_img = np.zeros((lens.col_img_uint.shape))
            new_col_img = lens.col_img_uint / 255.0
            lens.col_img = new_col_img
        else:
            lens.col_img = lens.col_img_uint

        lens.mask = mask
        lens.grid = local_grid
        
        # the lens area used for matching
        lens.inner_radius = calib.inner_lens_radius

        # minimum and maximum disparities for this lens
        focal_type = focal_type_map[_hex_focal_type(lc)]
        min_depth = calib.lens_types[focal_type]['depth_range']['min']
        max_depth = calib.lens_types[focal_type]['depth_range']['max']

        lens.min_disp = lens.diameter / max_depth
        lens.max_disp = lens.diameter / min_depth
        #pdb.set_trace()

        lenses[tuple(lc)] = lens

    return lenses


"""
Load an IMAGE, its DISPARITY and the CONFIDENCE (triplet)
Should be used with the gray-scaled version of the DISPARITY and the CONFIDENCE.
It is used for estimating a "conventional" disparity (see samples/disparity2D.py)
-----
November 2018
"""   
def load_triplet(img_filename, disp_name, conf_name, config_filename):   

    img = plt.imread(img_filename)
    disp = plt.imread(disp_name)
    conf = plt.imread(conf_name)
    calib = read_calibration(config_filename)
    
    if len(img.shape) < 3:
        print("This version works only for colored image")
        raise ValueError("Unsopported image dimension {0}".format(img.shape))    
    elif len(img.shape) == 3:
        weights = np.array([0.3, 0.59, 0.11])
        data = np.sum([img[:, :, i] * weights[i] for i in range(3)], axis=0)
        data_col = img
    else:
        raise ValueError("Unsupported image dimensions {0}".format(img.shape))
        
    # the image grid in pixels used for the bivariate spline interpolation
    gridy, gridx = range(data.shape[0]), range(data.shape[1])

    data_interp = sinterp.RectBivariateSpline(gridy, gridx, data)
    data_interp_r = sinterp.RectBivariateSpline(gridy, gridx, data_col[:,:,0])
    data_interp_g = sinterp.RectBivariateSpline(gridy, gridx, data_col[:,:,1])
    data_interp_b = sinterp.RectBivariateSpline(gridy, gridx, data_col[:,:,2])
    # we need to be using the gray version with the real values
    if disp[:,:,0].all() == disp[:,:,1].all() == disp[:,:,2].all():
        disp_data = disp[:,:,0]
    else:
        print("Wrong disparity map. Probably trying to read the colored version of the disparity map. Please use the gray version")
        raise ValueError("Wrong disparity map")    
    # we need to be using the gray version with the real values
    if conf[:,:,0].all() == conf[:,:,1].all() == conf[:,:,2].all():
        conf_data = conf[:,:,0]
    else:
        print("Wrong disparity map. Probably trying to read the colored version of the disparity map. Please use the gray version")
        raise ValueError("Wrong disparity map")    
    disp_interp = sinterp.RectBivariateSpline(gridy, gridx, disp_data)
    conf_interp = sinterp.RectBivariateSpline(gridy, gridx, conf_data)
    img_shape = np.asarray(data.shape[0:2])
    coords = rtxhexgrid.hex_lens_grid(img_shape, calib.lens_diameter, calib.rot_angle, calib.offset, calib.lbasis)
    focal_type_offsets = [tuple(lens['offset'].values()) for lens in calib.lens_types]
    focal_type_offsets = [(int(p[0]), int(p[1])) for p in focal_type_offsets]
    focal_type_map = {_hex_focal_type(p) : i for i, p in enumerate(focal_type_offsets)}

    lenses = dict()
    local_grid = rtxlens.LocalLensGrid(calib.lens_diameter)
    x, y = local_grid.x, local_grid.y
    xx, yy = local_grid.xx, local_grid.yy
    mask = np.zeros_like(local_grid.xx)
    mask[xx**2 + yy**2 < calib.inner_lens_radius**2] = 1    

    for lc in coords:

        pc = coords[lc]
        lens = rtxlens.Lens(lcenter=lc, pcenter=pc, diameter=calib.lens_diameter)
        lens.focal_type = _hex_focal_type(lc)
        lens.fstop = lens.diameter
        lens.pixel_size = 1.0
        lens.position = np.array([pc[1], pc[0], 0])
        cen_x = round(pc[0])
        cen_y = round(pc[1])
        x1 = int(cen_x + round(np.min(x)))
        x2 = int(cen_x + round(np.max(x)))
        y1 = int(cen_y + round(np.min(y)))
        y2 = int(cen_y + round(np.max(y)))
        lens.num_channels=3
        lens.img = data_interp(y + pc[0], x + pc[1])
        lens.disp_img = disp_interp(y + pc[0], x + pc[1])
        lens.conf_img = conf_interp(y + pc[0], x + pc[1])
        lens.img_interp = sinterp.RectBivariateSpline(y, x, lens.img)
        # colored version of the interpolated image:
        lens.img_interp3c[0] = data_interp_r
        lens.img_interp3c[1] = data_interp_g
        lens.img_interp3c[2] = data_interp_b
        lens.col_img_uint = data_col[x1:x2+1, y1:y2+1] #np.zeros((lens.img.shape[0], lens.img.shape[1], lens.num_channels))

        if lens.col_img_uint.dtype == 'uint8':
            new_col_img = np.zeros((lens.col_img_uint.shape))
            new_col_img = lens.col_img_uint / 255.0
            lens.col_img = new_col_img
        else:
            lens.col_img = lens.col_img_uint

        lens.mask = mask
        lens.grid = local_grid
        
        # the lens area used for matching
        lens.inner_radius = calib.inner_lens_radius

        # minimum and maximum disparities for this lens
        focal_type = focal_type_map[_hex_focal_type(lc)]
        min_depth = calib.lens_types[focal_type]['depth_range']['min']
        max_depth = calib.lens_types[focal_type]['depth_range']['max']

        lens.min_disp = lens.diameter / max_depth
        lens.max_disp = lens.diameter / min_depth
        #pdb.set_trace()

        lenses[tuple(lc)] = lens

    return lenses

"""
Load an image from the .xml file
-----
February 2018
"""  
def load_from_xml(image_filename, config_filename):

    img = plt.imread(image_filename)
    calib = read_calibration(config_filename)
    
    if len(img.shape) < 3:
        print("This version works only for colored image")
        raise ValueError("Unsopported image dimension {0}".format(img.shape))    
    elif len(img.shape) == 3:
        weights = np.array([0.3, 0.59, 0.11])
        data = np.sum([img[:, :, i] * weights[i] for i in range(3)], axis=0)
        data_col = img
    else:
        raise ValueError("Unsupported image dimensions {0}".format(img.shape))
        
    # the image grid in pixels used for the bivariate spline interpolation
    gridy, gridx = range(data.shape[0]), range(data.shape[1])

    data_interp = sinterp.RectBivariateSpline(gridy, gridx, data)
    data_interp_r = sinterp.RectBivariateSpline(gridy, gridx, data_col[:,:,0])
    data_interp_g = sinterp.RectBivariateSpline(gridy, gridx, data_col[:,:,1])
    data_interp_b = sinterp.RectBivariateSpline(gridy, gridx, data_col[:,:,2])
    img_shape = np.asarray(data.shape[0:2])
    coords = rtxhexgrid.hex_lens_grid(img_shape, calib.lens_diameter, calib.rot_angle, calib.offset, calib.lbasis)
    focal_type_offsets = [tuple(lens['offset'].values()) for lens in calib.lens_types]
    focal_type_offsets = [(int(p[0]), int(p[1])) for p in focal_type_offsets]
    focal_type_map = {_hex_focal_type(p) : i for i, p in enumerate(focal_type_offsets)}
    
    lenses = dict()
    local_grid = rtxlens.LocalLensGrid(calib.lens_diameter)
    x, y = local_grid.x, local_grid.y
    xx, yy = local_grid.xx, local_grid.yy
    mask = np.zeros_like(local_grid.xx)
    mask[xx**2 + yy**2 < calib.inner_lens_radius**2] = 1    
    
    for lc in coords:

        pc = coords[lc]
        lens = rtxlens.Lens(lcenter=lc, pcenter=pc, diameter=calib.lens_diameter)
        lens.focal_type = _hex_focal_type(lc)
        lens.fstop = lens.diameter
        lens.pixel_size = 1.0
        lens.position = np.array([pc[1], pc[0], 0])
        lens.img = np.ones((local_grid.shape[0], local_grid.shape[1], 3))

        cen_x = round(pc[0])
        cen_y = round(pc[1])
        x1 = int(cen_x + round(np.min(x)))
        x2 = int(cen_x + round(np.max(x)))
        y1 = int(cen_y + round(np.min(y)))
        y2 = int(cen_y + round(np.max(y)))
        lens.num_channels=3
        lens.img = data_interp(y + pc[0], x + pc[1])
        lens.img_interp = sinterp.RectBivariateSpline(y, x, lens.img)
        # colored version of the interpolated image:
        lens.img_interp3c[0] = data_interp_r
        lens.img_interp3c[1] = data_interp_g
        lens.img_interp3c[2] = data_interp_b
        lens.col_img_uint = data_col[x1:x2+1, y1:y2+1] #np.zeros((lens.img.shape[0], lens.img.shape[1], lens.num_channels))

        if lens.col_img_uint.dtype == 'uint8':
            new_col_img = np.zeros((lens.col_img_uint.shape))
            new_col_img = lens.col_img_uint / 255.0
            lens.col_img = new_col_img
        else:
            lens.col_img = lens.col_img_uint

        lens.mask = mask
        lens.grid = local_grid
        
        # the lens area used for matching
        lens.inner_radius = calib.inner_lens_radius

        # minimum and maximum disparities for this lens
        focal_type = focal_type_map[_hex_focal_type(lc)]
        min_depth = calib.lens_types[focal_type]['depth_range']['min']
        max_depth = calib.lens_types[focal_type]['depth_range']['max']

        lens.min_disp = lens.diameter / max_depth
        lens.max_disp = lens.diameter / min_depth
        #pdb.set_trace()

        lenses[tuple(lc)] = lens

    return lenses    

"""
Load a synthetic image (actually a lot of micro-iamges) from the respective .json file
-----
February 2018
"""     
def load_from_json(filename):

    """
    Loads a synthetic MLA dataset

    Parameters:
    -----------

    filename: string
              Filename of the scene configuration file (format: JSON)
rtxrender
    Returns:
    --------

    lenses: dictionary
              Lens dictionary, keys are the axial coordinates
    """
    
    basedir = os.path.dirname(filename)

    with open(filename, 'r') as f:
        lenses_json = json.load(f)

    lenses = dict()

    if len(lenses_json) == 0:
        return lenses

    diam = lenses_json[0]['diameter']
    radius = diam / 2.0
    inner_radius = radius - lenses_json[0]['lens_border']
    local_grid = rtxlens.LocalLensGrid(diam)
  
    # local grid coordinates shared by all lenses
    x, y = local_grid.x, local_grid.y
    xx, yy = local_grid.xx, local_grid.yy
    mask = np.zeros_like(local_grid.xx)
    mask[xx**2 + yy**2 < inner_radius**2] = 1
    
    for lens in lenses_json:
        # image data
        color_filename = "{0}/{1}".format(basedir, lens['relative_color_filename'])

        # ground truth depth
        depth_filename = "{0}/{1}".format(basedir, lens['relative_depth_filename'])
        
        tmp_lens = rtxlens.Lens(lens['axial_coord'], lens['pixel_coord'], lens['diameter'], lens['focal_type'])
        
        tmp_lens.col_img = plt.imread(color_filename)
        if tmp_lens.col_img.shape[2] == 4:
          tmp_lens.col_img = tmp_lens.col_img[:,:,:3]
        weights = np.array([0.3, 0.59, 0.11])
        tmp_lens.img = np.sum([tmp_lens.col_img[:, :, i].astype(np.float64) * weights[i] for i in range(3)], axis=0)
        tmp_lens.grid = local_grid
        tmp_lens.mask = mask
        tmp_lens.num_channels = tmp_lens.col_img.shape[2]
        tmp_lens.img_interp = sinterp.RectBivariateSpline(y, x, tmp_lens.img)
        tmp_lens.img_interp3c[0] = sinterp.RectBivariateSpline(y, x, tmp_lens.col_img[:,:,0])
        tmp_lens.img_interp3c[1] = sinterp.RectBivariateSpline(y, x, tmp_lens.col_img[:,:,1])
        tmp_lens.img_interp3c[2] = sinterp.RectBivariateSpline(y, x, tmp_lens.col_img[:,:,2])
        tmp_lens.inner_radius = inner_radius

        tmp_lens.focal_length = lens['focal_length']
        tmp_lens.focus_distance = lens['focus_distance']
        tmp_lens.fstop = lens['fstop']
        tmp_lens.pixel_size = lens['pixel_size']
        tmp_lens.position = np.array(lens['position'])  
        #tmp_lens.position = np.array(lens['location'])  # some datasets needs 'location'
        tmp_lens.rotation = np.array(lens['rotation_mat']).reshape((3, 3)) 
        #tmp_lens.rotation = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).reshape((3,3)) #
        
        tmp_lens.focus_disparity = tmp_lens.focal_length * tmp_lens.diameter / tmp_lens.focus_distance
        
        # read the ground truth depth. The 16-Bit PNG depth files
        # contain the clipped depth values, convert to the real depth
        # values        
        tmp_depth = plt.imread(depth_filename).astype(np.float64) 
        tmp_depth = tmp_depth * (lens['clip_end'] - lens['clip_start']) + lens['clip_start']
        tmp_lens.depth_img = tmp_depth
        tmp_lens.disp_img = np.copy(tmp_depth)
        tmp_lens.disp_img[tmp_depth > 0] = (tmp_lens.focal_length * tmp_lens.diameter) / tmp_lens.disp_img[tmp_depth > 0]

        # set the lens border
        tmp_lens.lens_border = lens['lens_border']

        # finally add the lens to the dictionary
        lenses[tuple(lens['axial_coord'])] = tmp_lens

    return lenses    

def save_only_xml(filename, img_shape, central_lens, lens_border, angle):

    h,w = img_shape[0], img_shape[1]
    m = ((h-1)/2.0, (w-1) / 2.0)
    offset = central_lens.pcoord - np.array(m)
    config_template = _xml_template()
    config = config_template.substitute(offset_y=offset[0], offset_x=offset[1], diam=central_lens.diameter,
                                        angle=angle, lens_border=lens_border)
    config_filename = (filename)
    with open(config_filename, "w") as f:
        f.write(config)

    return config
    
def save_xml(filename, lenses):

    """
    Saves a synthetic dataset in the Raytrix Format: PNG Image file + xml config
    
    Attention: The PNG File is saved as a 4 channel PNG

    Parameters
    ----------

    filename: string
      The filename for the target files. If filename
      is <path>/file the target files are <path>/file.png, <path>/file.xml

    lenses: dictionary
      Dictionary of lenses, keys are axial coordinates
      
    """

    # extract the images as a seperate dictionary for the rendering process
    lens_data = dict()
    disp_data = dict()
    for lcoord in lenses:
        lens_data[lcoord] = lenses[lcoord].col_img
        disp_data[lcoord] = lenses[lcoord].disp_img
        
    # render the MLA image
    img = rtxrnd.render_lens_imgs(lenses, lens_data)
    disp = rtxrnd.render_lens_imgs(lenses, disp_data)

    h, w = img.shape[0], img.shape[1]
    m = ((h-1)/2.0, (w-1) / 2.0)
    offset = lenses[0, 0].pcoord - np.array(m)

    l = lenses[0, 0]
    config_template = _xml_template()
    config = config_template.substitute(offset_y=offset[0], offset_x=offset[1], diam=l.diameter,
                                        angle=0, lens_border=1.0)

    img_filename = "{0}.png".format(filename)
    disp_filename = "{0}_disp.png".format(filename)
    config_filename = "{0}.xml".format(filename)

    with open(config_filename, "w") as f:
        f.write(config)

    plt.imsave(img_filename, img)
    plt.imsave(disp_filename, disp, cmap='jet')

    return config
    
def _xml_template():
    s = string.Template('''<RayCalibData version="1.0">
  <offset units="pix">
    <x>$offset_x</x>
    <y>$offset_y</y>
  </offset>
  <diameter units="pix">$diam</diameter>
  <rotation units="rad">$angle</rotation>
  <lens_border units="pix">$lens_border</lens_border>
  <tcp units="virtual_depth">2.000000000000</tcp>
  <lens_base_x units="lens">
    <x>1.000000000000</x>
    <y>0.000000000000</y>
  </lens_base_x>
  <lens_base_y units="lens">
    <x>0.500000000000</x>
    <y>0.866025403784</y>
  </lens_base_y>
  <sub_grid_base units="lens">
    <x>3.000000000000</x>
    <y>1.732050807569</y>
  </sub_grid_base>
  <distortion type="radial">
    <function>constant</function>
    <parameter count="0">0</parameter>
    <offset units="pix">
      <x>0.000000000000</x>
      <y>0.000000000000</y>
    </offset>
  </distortion>
  <lens_type id="0">
    <offset units="lens">
      <x>0.000000000000</x>
      <y>0.000000000000</y>
    </offset>
    <depth_range units="virtual_depth">
      <min>1.000000000000</min>
      <max>3.000000000000</max>
    </depth_range>
  </lens_type>
  <lens_type id="1">
    <offset units="lens">
      <x>1.000000000000</x>
      <y>0.000000000000</y>
    </offset>
    <depth_range units="virtual_depth">
      <min>2.799999952316</min>
      <max>4.000000000000</max>
    </depth_range>
  </lens_type>
  <lens_type id="2">
    <offset units="lens">
      <x>-1.000000000000</x>
      <y>0.000000000000</y>
    </offset>
    <depth_range units="virtual_depth">
      <min>3.799999952316</min>
      <max>100.000000000000</max>
    </depth_range>
  </lens_type>
</RayCalibData>
''')
    return s
    
def write_csv_file(error_analysis, err_analysis_csv_name, technique):
    
    file = open(err_analysis_csv_name,'w')
    file.write("CSV file with ,ERROR MEASUREMENTS ,for ,{0} ,technique\n".format(technique))
    file.write("\n")
    file.write(", , ,0,1,2\n")
    file.write("{0},BadPix1, , , , ,{1}\n".format(technique, error_analysis['badpix1_avg']))
    file.write(",BadPix2, , , , ,{0}\n".format(error_analysis['badpix2_avg']))
    file.write(",AvgErr, ,{0},{1},{2}\n".format(error_analysis['avg_error'][0]['err'], error_analysis['avg_error'][1]['err'], error_analysis['avg_error'][2]['err']))
    file.write(",StdDevAvg, ,{0},{1},{2}\n".format(error_analysis['avg_error'][0]['std'], error_analysis['avg_error'][1]['std'], error_analysis['avg_error'][2]['std']))
    file.write(",MSE, ,{0},{1},{2}\n".format(error_analysis['mse_error'][0]['err'], error_analysis['mse_error'][1]['err'], error_analysis['mse_error'][2]['err']))
    file.write(",StdDevMSE, ,{0},{1},{2}\n".format(error_analysis['mse_error'][0]['std'], error_analysis['mse_error'][1]['std'], error_analysis['mse_error'][2]['std']))
    file.write(",Bumpiness, ,{0},{1},{2}\n".format(error_analysis['bumpiness'][0]['err'], error_analysis['bumpiness'][1]['err'], error_analysis['bumpiness'][2]['err']))
    file.write(",StdDevBump, ,{0},{1},{2}\n".format(error_analysis['bumpiness'][0]['std'], error_analysis['bumpiness'][1]['std'], error_analysis['bumpiness'][2]['std']))
    file.write(",AvgErrDisc, ,{0},{1},{2}\n".format(error_analysis['disc_err'][0]['err'], error_analysis['disc_err'][1]['err'], error_analysis['disc_err'][2]['err']))
    file.write(",StdDevDisc, ,{0},{1},{2}\n".format(error_analysis['disc_err'][0]['std'], error_analysis['disc_err'][1]['std'], error_analysis['disc_err'][2]['std']))
    file.write(",AvgErrSmooth, ,{0},{1},{2}\n".format(error_analysis['smooth_err'][0]['err'], error_analysis['smooth_err'][1]['err'], error_analysis['smooth_err'][2]['err']))
    file.write(",StdDevSmooth, ,{0},{1},{2}\n".format(error_analysis['smooth_err'][0]['std'], error_analysis['smooth_err'][1]['std'], error_analysis['smooth_err'][2]['std']))
    file.write(",BadPix1Disc, , , , ,{0}\n".format(error_analysis['badpix1disc']))
    file.write(",BadPix1Smooth, , , , ,{0}\n".format(error_analysis['badpix1smooth']))  
    file.write(",BadPix2Disc, , , , ,{0}\n".format(error_analysis['badpix2disc']))
    file.write(",BadPix2Smooth, , , , ,{0}\n".format(error_analysis['badpix2smooth']))   
    
def write_csv_array(arrays, name, technique):

    err=arrays[0]
    disc=arrays[1]
    smth=arrays[2]
    file = open(name,'w')
    file.write("CSV file with ,NUMBER OF WRONG PIXELS, against ,ERROR THRESHOLD ,for ,{0} ,technique\n".format(technique))
    file.write("\n")
    file.write("Error's Thresholds, ,")
    for satana in range(len(err)):
        file.write("{0},".format(0.1*satana))
    file.write("\n")
    file.write("Wrong Pixels, ,")
    for satana in range(len(err)):
        file.write("{0},".format(err[satana]))
    file.write("\n")  
    file.write("Wrong Pixels Disc, ,")
    for satana in range(len(disc)):
        file.write("{0},".format(disc[satana]))
    file.write("\n")     
    file.write("Wrong Pixels Smooth, ,")
    for satana in range(len(smth)):
        file.write("{0},".format(smth[satana]))
    file.write("\n")    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
