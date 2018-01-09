import tkinter as tk
from tkinter import *
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.backends.tkagg as tkagg
#from matplotlib.backends.backend_agg import FigureCanvasAgg
#from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
import pdb
import numpy as np
import plenopticIO.imgIO as plIO
import disparity.disparity_methods as pldisp
import os

 
# read the image with PhotoIMage
img_1 = None
img_2 = None
img_3 = None
img_4 = None

# resized version (need to keep a reference with tkinter)
img_1_display = None
img_2_display = None
img_3_display = None
img_4_display = None
img_1_subview = None
img_2_subview = None
img_3_subview = None
img_4_subview = None

# real image on which we can estimate depth usw
img_1_dict = None
img_2_dict = None
img_3_dict = None
img_4_dict = None

# variable to know if we loaded images
isThereImg1 = False
isThereImg2 = False
isThereImg3 = False
isThereImg4 = False

# height and wdht of pictures
width_canvas = 1920
height_canvas = 1080

class Gui():
    def __init__(self, root):
        self.root=root
        self.root.title("Plenoptic 2.0 Toolbox")
        self.entry = tk.Entry(root)
        global num_img, img_target, img_target2, disp_target, disp_store, method, folder_path
        #parameters
        global minValue, maxValue, cocValue, pen1Value, pen2Value, coverValue
        
        num_img=tk.StringVar()
        num_img.set("Img1")
        num_img.trace("w", option_changed)
        folder_path = tk.StringVar()
        root.canvas=tk.Canvas(root, width=width_canvas, height=height_canvas, background='white')
        root.canvas.grid(row=0,column=1)

        frame = Frame(self.root)
        frame.grid(row=0,column=0, sticky="n")
        
        # IMAGES PART
        labelImg=Label(frame, text="Images", font=("Helvetica", 16)).grid(row=0,column=0, sticky="w")
        label1=Label(frame, text="Display Figure").grid(row=1,column=0, sticky="nw")
        self.option=tk.OptionMenu(frame, num_img, "Img1", "Img2", "Img3", "Img4", "All")
        self.option.grid(row=1,column=1,sticky="nwe")
        folderlab = Label(frame, text="Folder").grid(row=2,column=0, sticky="w")
        label2=Label(frame, text="Figure1").grid(row=3,column=0, sticky="w")
        label3=Label(frame, text="Figure2").grid(row=4,column=0, sticky="w")
        label4=Label(frame, text="Figure3").grid(row=5,column=0, sticky="w")
        label5=Label(frame, text="Figure4").grid(row=6,column=0, sticky="w")
        root.folderentry = Entry(frame)
        root.folderentry.insert(END, '/data1/palmieri/PlenopticToolbox/TestImages')
        root.folderentry.grid(row = 2,column = 1,sticky = E+ W)
        root.entry = Entry(frame)
        root.entry.insert(END, 'Cards.png')
        root.entry.grid(row = 3,column = 1,sticky = E+ W)
        root.entry1 = Entry(frame)
        root.entry1.insert(END, 'Hawaii.png')
        root.entry1.grid(row = 4,column = 1, sticky = E)
        root.entry2 = Entry(frame)
        root.entry2.insert(END, 'Specular.png')
        root.entry2.grid(row = 5,column = 1,sticky = E+ W)
        root.entry3 = Entry(frame)
        root.entry3.insert(END, 'Trucks.png')
        root.entry3.grid(row = 6,column = 1, sticky = E)
        ButtonFolder=Button(frame,text="Change Folder", command=change_path).grid(row = 2,column = 2, sticky = "we")
        Button1=Button(frame,text="Load", command=loadimg1).grid(row = 3,column = 2, sticky = "we")
        Button2=Button(frame,text="Load", command=loadimg2).grid(row = 4,column = 2, sticky = "we")
        Button3=Button(frame,text="Load", command=loadimg3).grid(row = 5,column = 2, sticky = "we")
        Button4=Button(frame,text="Load", command=loadimg4).grid(row = 6,column = 2, sticky = "we")
        
        
        # DISPARITY PART
        labelImg=Label(frame, text="Disparity Estimation", font=("Helvetica", 16)).grid(row=7,column=0, sticky="w")
        
        img_target=tk.StringVar()
        img_target.set("Img1")
        method=tk.StringVar()
        method.set("SAD")
        disp_store = tk.StringVar()
        disp_store.set("Img2")
                
        to_whom=Label(frame, text="Input Image?").grid(row=8,column=0, sticky="nw")
        which_met=Label(frame, text="Save As?").grid(row=8,column=1, sticky="nw")
        self.option=tk.OptionMenu(frame, img_target, "Img1", "Img2", "Img3", "Img4")
        self.option.grid(row=9,column=0,sticky="nwe")
        root.option2 = Entry(frame)
        root.option2.insert(END, 'Disparity.png')
        root.option2.grid(row=9,column=1,sticky="nwe")
        Button1=Button(frame,text="Estimate", command=estimatedisp).grid(row = 9,column = 2, sticky = "we")
                
                
        parameterLabel=Label(frame, text="Parameters",  font=("Helvetica", 13)).grid(row=10,column=0, sticky="w")
        store_where=Label(frame, text="Similarity Measure").grid(row=11,column=0, sticky="nw")
        self.option=tk.OptionMenu(frame, method, "SAD", "SSD", "Census")
        self.option.grid(row=11,column=1 ,sticky="nwe")
        minLab = Label(frame, text="Minimum Disparity").grid(row=12, column=0, sticky="w")
        maxLab = Label(frame, text="Maximum Disparity").grid(row=13, column=0, sticky="w")
        cocLab = Label(frame, text="Circle of Confusion").grid(row=14, column=0, sticky="w")
        pen1 = Label(frame, text="Penalty 1").grid(row=15, column=0, sticky="w")
        pen2 = Label(frame, text="Penalty 2").grid(row=16, column=0, sticky="w")
        cover = Label(frame, text="Maximal Baseline").grid(row=17, column=0, sticky="w")
        root.minValue = Entry(frame)
        root.minValue.insert(END, '0')
        root.minValue.grid(row =12,column = 1,sticky = E+ W)
        root.maxValue = Entry(frame)
        root.maxValue.insert(END, '12')
        root.maxValue.grid(row =13,column = 1,sticky = E+ W)
        root.cocValue = Entry(frame)
        root.cocValue.insert(END, '1.5')
        root.cocValue.grid(row =14,column = 1,sticky = E+ W)
        root.pen1Value = Entry(frame)
        root.pen1Value.insert(END, '0.1')
        root.pen1Value.grid(row =15,column = 1,sticky = E+ W)
        root.pen2Value = Entry(frame)
        root.pen2Value.insert(END, '0.03')
        root.pen2Value.grid(row =16,column = 1,sticky = E+ W)
        root.coverValue = Entry(frame)
        root.coverValue.insert(END, '7')
        root.coverValue.grid(row =17,column = 1,sticky = E+ W)        
        
        #ALLINFOCUS
        labelImg=Label(frame, text="All In Focus Images", font=("Helvetica", 16)).grid(row=19,column=0, sticky="w")
        
        img_target2=tk.StringVar()
        img_target2.set("Img1")
        disp_target=tk.StringVar()
        disp_target.set("Img2")

        to_whom=Label(frame, text="Color Image?").grid(row=20,column=0, sticky="nw")
        which_met=Label(frame, text="Disparity Image?").grid(row=20,column=1, sticky="nw")
        self.option=tk.OptionMenu(frame, img_target2, "Img1", "Img2", "Img3", "Img4")
        self.option.grid(row=21,column=0,sticky="nwe")
        self.option=tk.OptionMenu(frame, disp_target, "Img1", "Img2", "Img3", "Img4")
        self.option.grid(row=21,column=1 ,sticky="nwe")
        Button1=Button(frame,text="Compute").grid(row = 21,column = 2, sticky = "we")    
        
        
        #FILLING, FILTERING
        labelImg=Label(frame, text="Filling", font=("Helvetica", 16)).grid(row=25,column=0, sticky="w")
        
        fig, ax = plt.subplots(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')
        #pdb.set_trace()
        #print(get_path(self))
        #print(num_img.get())
        
        refresh_images()
    def get_path(self):
        #pdb.set_trace()
        return root.folderentry.get()
    def get_entry1(self):
        return root.entry.get()
    def get_entry2(self):
        return root.entry1.get()
    def get_entry3(self):
        return root.entry2.get()
    def get_entry4(self):
        return root.entry3.get()   
    def get_dmin(self):
        return root.minValue.get()
    def get_dmax(self):
        return root.maxValue.get()
    def get_coc(self):
        return root.cocValue.get()        
    def get_pen1(self): 
        return root.pen1Value.get()   
    def get_pen2(self):
        return root.pen2Value.get()       #Grid.columnconfigure(self.root,1,weight=1, size=200)
    def get_cover(self):
        return root.coverValue.get()   
    def get_disp_path(self):
        return root.option2.get()
            
def change_path():

    print("Folder changed to {0}".format(Gui.get_path(root)))

def load_real_img(image_path):

    config_filename = image_path.split(".")[0] + ".xml"
    real_img = plIO.load_from_xml(image_path, config_filename)
    return real_img

def loadimg1():

    image_path = "{0}/{1}".format(Gui.get_path(root), Gui.get_entry1(root))
    print("Loading image {0}".format(image_path))
    #pdb.set_trace()
    global img_1 #, img_1_dict
    #img_1 = plt.imread(image_path)
    img_1 = PhotoImage(file = image_path)
    #img_1_dict = load_real_img(image_path)
    global isThereImg1
    isThereImg1 = True

    #img_1 = plt.imread(image_path)
    if isThereImg1:
        print("All right! Image 1 Loaded")
    refresh_images()
    
def loadimg2():

    image_path = "{0}/{1}".format(Gui.get_path(root), Gui.get_entry2(root))
    print("Loading image {0}".format(image_path))
    #pdb.set_trace()
    global img_2
    #img_1 = plt.imread(image_path)
    img_2 = PhotoImage(file = image_path)
    global isThereImg2
    isThereImg2 = True

    #img_1 = plt.imread(image_path)
    if isThereImg2:
        print("All right! Image 2 Loaded")
    refresh_images()
       
def loadimg3():

    image_path = "{0}/{1}".format(Gui.get_path(root), Gui.get_entry3(root))
    print("Loading image {0}".format(image_path))
    #pdb.set_trace()
    global img_3
    #img_1 = plt.imread(image_path)
    img_3 = PhotoImage(file = image_path)
    global isThereImg3
    isThereImg3 = True

    #img_1 = plt.imread(image_path)
    if isThereImg3:
        print("All right! Image 3 Loaded")
    refresh_images()
    
def loadimg4():

    image_path = "{0}/{1}".format(Gui.get_path(root), Gui.get_entry4(root))
    print("Loading image {0}".format(image_path))
    #pdb.set_trace()
    global img_4
    #img_1 = plt.imread(image_path)
    img_4 = PhotoImage(file = image_path)
    global isThereImg4
    isThereImg4 = True

    #img_1 = plt.imread(image_path)
    if isThereImg4:
        print("All right! Image 4 Loaded")
    refresh_images()
           
def refresh_images(config='nochange'):
    
    if config == 'nochange':
        #check num_img variable
        if num_img.get() == 'Img1':
        #pdb.set_trace()
            root.canvas.delete("all")
            #pdb.set_trace()
            if isThereImg1:
                #pdb.set_trace()
                #print('ciao')
                #fig = plt.figure()
                #fig.add_subplot(111)
                #im = fig2img(fig)
                #test = root.canvas.create_image(0, 0, image=im, anchor='nw')
                #pdb.set_trace()
                global img_1_display
                if img_1.width() > width_canvas:
                    factor = img_1.width() / width_canvas
                    int_factor = int(np.ceil(factor))
                    img_1_display = img_1.subsample(int_factor)
                else:
                    img_1_display = img_1
                x_pos = int((width_canvas - img_1_display.width() ) / 2)
                y_pos = int((height_canvas - img_1_display.height() ) / 2)
                #pdb.set_trace()
                root.canvas.create_image(x_pos, y_pos, image=img_1_display, anchor="nw")
            else:
                root.canvas.create_rectangle(0, 0, width_canvas, height_canvas, fill='green')
                canvas_id = root.canvas.create_text(int(width_canvas/2), int(height_canvas/2), anchor="nw")
                root.canvas.insert(canvas_id, 50, "IMG1")    
        elif num_img.get() == 'Img2':
        #pdb.set_trace()
            root.canvas.delete("all")
            if isThereImg2:
                global img_2_display
                if img_2.width() > width_canvas:
                    factor = img_2.width() / width_canvas
                    int_factor = int(np.ceil(factor))
                    img_2_display = img_2.subsample(int_factor)
                else:
                    img_2_display = img_2
                x_pos = int((width_canvas - img_2_display.width() ) / 2)
                y_pos = int((height_canvas - img_2_display.height() ) / 2)
                #pdb.set_trace()
                root.canvas.create_image(x_pos, y_pos, image=img_2_display, anchor="nw")
            else:
                root.canvas.create_rectangle(0, 0, width_canvas, height_canvas, fill='red')
            canvas_id = root.canvas.create_text(int(width_canvas/2), int(height_canvas/2), anchor="nw")
            root.canvas.insert(canvas_id, 50, "IMG2")
        elif num_img.get() == 'Img3':
        #pdb.set_trace()
            root.canvas.delete("all")
            if isThereImg3:
                global img_3_display
                if img_3.width() > width_canvas:
                    factor = img_3.width() / width_canvas
                    int_factor = int(np.ceil(factor))
                    img_3_display = img_3.subsample(int_factor)
                else:
                    img_3_display = img_3
                x_pos = int((width_canvas - img_3_display.width() ) / 2)
                y_pos = int((height_canvas - img_3_display.height() ) / 2)
                #pdb.set_trace()
                root.canvas.create_image(x_pos, y_pos, image=img_3_display, anchor="nw")
            else:
                root.canvas.create_rectangle(0, 0, width_canvas, height_canvas, fill='blue')
            canvas_id = root.canvas.create_text(int(width_canvas/2), int(height_canvas/2), anchor="nw")
            root.canvas.insert(canvas_id, 50, "IMG3")
        elif num_img.get() == 'Img4':
        #pdb.set_trace()
            root.canvas.delete("all")
            if isThereImg4:
                global img_4_display
                if img_4.width() > width_canvas:
                    factor = img_4.width() / width_canvas
                    int_factor = int(np.ceil(factor))
                    img_4_display = img_4.subsample(int_factor)
                else:
                    img_4_display = img_4
                x_pos = int((width_canvas - img_4_display.width() ) / 2)
                y_pos = int((height_canvas - img_4_display.height() ) / 2)
                #pdb.set_trace()
                root.canvas.create_image(x_pos, y_pos, image=img_4_display, anchor="nw")
            else:
                root.canvas.create_rectangle(0, 0, width_canvas, height_canvas, fill='yellow')
            canvas_id = root.canvas.create_text(int(width_canvas/2), int(height_canvas/2), anchor="nw")
            root.canvas.insert(canvas_id, 50, "IMG4")
            #figure1=self.canvas.create_rectangle(80, 80, 120, 120, fill="blue")
        
        elif num_img.get() == 'All':
            #pdb.set_trace()
            root.canvas.delete("all")
            height_canvas_small = int(height_canvas / 2)
            width_canvas_small = int(width_canvas / 2)
            root.canvas.create_line(0, height_canvas_small, width_canvas, height_canvas_small, fill="black", width=2.0)
            root.canvas.create_line(width_canvas_small, 0, width_canvas_small, height_canvas, fill="black", width=2.0)
            if isThereImg1:
                global img_1_subview
                #pdb.set_trace()
                factor = resizing_factor(img_1, [width_canvas_small, height_canvas_small])
                if factor > 1:
                    img_1_subview = img_1.subsample(factor)
                else:
                    img_1_subview = img_1
                x_pos = int((width_canvas_small - img_1_subview.width() ) / 2)
                y_pos = int((height_canvas_small - img_1_subview.height() ) / 2)
                #pdb.set_trace()
                root.canvas.create_image(x_pos, y_pos, image=img_1_subview, anchor="nw")
            else:
                #root.canvas.create_rectangle(0, 0, width_canvas_small, height_canvas_small, fill='green')
                canvas_id = root.canvas.create_text(int(height_canvas_small/2), int(width_canvas_small/2), anchor="nw")
                root.canvas.insert(canvas_id, 20, "IMG1") 
            if isThereImg2:
                global img_2_subview
                factor = resizing_factor(img_2, [width_canvas_small, height_canvas_small])
                if factor > 1:
                    img_2_subview = img_2.subsample(factor)
                else:
                    img_2_subview = img_2
                x_pos = int((width_canvas_small - img_2_subview.width() ) / 2)
                y_pos = int((height_canvas_small - img_2_subview.height() ) / 2)
                #pdb.set_trace()
                root.canvas.create_image(width_canvas_small + x_pos, y_pos, image=img_2_subview, anchor="nw")
            else:
                #root.canvas.create_rectangle(width_canvas_small, 0, width_canvas_small, height_canvas_small, fill='red')
                canvas_id = root.canvas.create_text(int(width_canvas_small/2), int(height_canvas_small/2), anchor="nw")
                root.canvas.insert(canvas_id, 20, "IMG2")        
            if isThereImg3:
                global img_3_subview
                factor = resizing_factor(img_3, [width_canvas_small, height_canvas_small])
                if factor > 1:
                    img_3_subview = img_3.subsample(factor)
                else:
                    img_3_subview = img_3
                x_pos = int((width_canvas_small - img_3_subview.width() ) / 2)
                y_pos = int((height_canvas_small - img_3_subview.height() ) / 2)
                #pdb.set_trace()
                root.canvas.create_image(x_pos, height_canvas_small + y_pos, image=img_3_subview, anchor="nw")
            else:
                #root.canvas.create_rectangle(0, height_canvas_small, width_canvas_small, height_canvas_small, fill='blue')
                canvas_id = root.canvas.create_text(int(width_canvas_small/2), int(height_canvas_small/2), anchor="nw")
                root.canvas.insert(canvas_id, 20, "IMG3")
            if isThereImg4:
                global img_4_subview
                factor = resizing_factor(img_4, [width_canvas_small, height_canvas_small])
                if factor > 1:
                    img_4_subview = img_4.subsample(factor)
                else:
                    img_4_subview = img_4
                x_pos = int((width_canvas_small - img_4_subview.width() ) / 2)
                y_pos = int((height_canvas_small - img_4_subview.height() ) / 2)
                #pdb.set_trace()
                root.canvas.create_image(width_canvas_small + x_pos, height_canvas_small + y_pos, image=img_4_subview, anchor="nw")
            else:
                #root.canvas.create_rectangle(width_canvas_small, height_canvas_small, width_canvas_small, height_canvas_small, fill='yellow')
                canvas_id = root.canvas.create_text(int(width_canvas_small/2), int(height_canvas_small/2), anchor="nw")
                root.canvas.insert(canvas_id, 20, "IMG4")
            #pdb.set_trace()
    
    return True

def resizing_factor(img, canvas_size):

    if img.width() < canvas_size[0] and img.width() < canvas_size[1]:
        return 1
        
    height_factor = img.height() / canvas_size[1]
    int_height_factor = int(np.ceil(height_factor))
    width_factor = img.width() / canvas_size[0]
    int_width_factor = int(np.ceil(width_factor))
    
    return max(int_height_factor, int_width_factor)

def fig2img ( fig ):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data ( fig )
    w, h, d = buf.shape
    return Image.fromstring( "RGBA", ( w ,h ), buf.tostring( ) )

def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.frombytes( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf

def estimatedisp():
        
    # check img source and target
    img_source = img_target.get()
    img_store = disp_store.get()
    if img_source == 'Img1':
        if not isThereImg1:
            print("Missing the input image! Please load img1")
            #return
        else:
            # read the parameters
            min_disp = float(Gui.get_dmin(root))
            if min_disp < 0 or min_disp > 12:
                print("Wrong value of Minimum Disparity!")
            max_disp = float(Gui.get_dmax(root))
            coc = Gui.get_coc(root)
            pen1 = Gui.get_pen1(root)
            pen2 = Gui.get_pen2(root)
            cover = Gui.get_cover(root)
            image_path_png = "{0}/{1}".format(Gui.get_path(root), Gui.get_entry1(root))
            basename, suffix = os.path.splitext(image_path_png)
            image_path_config = basename + ".xml"
            #pack all
            params = pldisp.EvalParameters()
            params.filename = image_path_config
            params.scene_type = 'real'
            params.min_disp = float(min_disp)
            params.max_disp = float(max_disp)
            params.coc_thresh = float(coc)
            params.penalty1 = float(pen1)
            params.penalty2 = float(pen2)
            params.max_ring = float(cover)
            
            disp_pack = pldisp.estimate_disp(params)
            pdb.set_trace()
    if img_source == 'Img2' and not isThereImg2:
        print("Missing the input image! Please load img2")
        return
    if img_source == 'Img3' and not isThereImg3:
        print("Missing the input image! Please load img3")
        return
    if img_source == 'Img4' and not isThereImg4:
        print("Missing the input image! Please load img4")
        return
    
    disp_path_to_save = "{0}/{1}".format(Gui.get_path(root), Gui.get_disp_path(root))
    plt.imsave(disp_path_to_save, disp_pack[1])
    print("Disparity Calculated and Saved!")
    #done = refresh_images('nochange')
            
def option_changed(*args):

    #pdb.set_trace()
    # IMAGES FAKE        
    # IMAGES
    #canvas.delete("all")
    done = refresh_images('nochange')
    #figure1=self.canvas.create_rectangle(80, 80, 123, 120, fill="blue")
    
    print(num_img.get())
    
#def loadimage1():        
def draw_figure(canvas, figure, loc=(0, 0)):
    """ 
    Draw a matplotlib figure onto a Tk canvas

    loc: location of top-left corner of figure on canvas in pixels.
    Inspired by matplotlib source: lib/matplotlib/backends/backend_tkagg.py
    """
    figure_canvas_agg = FigureCanvasAgg(figure)
    figure_canvas_agg.draw()
    figure_x, figure_y, figure_w, figure_h = figure.bbox.bounds
    figure_w, figure_h = int(figure_w), int(figure_h)
    photo = tk.PhotoImage(master=canvas, width=figure_w, height=figure_h)

    # Position: convert from top-left anchor to center anchor
    canvas.create_image(loc[0] + figure_w/2, loc[1] + figure_h/2, image=photo)

    # Unfortunately, there's no accessor for the pointer to the native renderer
    tkagg.blit(photo, figure_canvas_agg.get_renderer()._renderer, colormode=2)

    # Return a handle which contains a reference to the photo object
    # which must be kept live or else the picture disappears
    return photo
    
def ok():
    print("value is", stvar.get())
    #root.quit()

if __name__== '__main__':
    root=tk.Tk()
    gui=Gui(root)
    #root.mainloop()
    while True:
        root.update_idletasks()
        root.update()
