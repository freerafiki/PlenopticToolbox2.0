import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import pdb
"""
Read a set of SI generated and create a huge image as if it was a synthetically generated lenslet image
This is used to get the input image for the SPO code in Matlab
It can also output EPI_h and EPI_v, just use the parameters --epi_h and --epi_v when launching.

The drawback of this approach is that we need the set of SI generated using the disparity.
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Read an image and estimate disparity")
    parser.add_argument(dest='data_folder', nargs=1, help="Name of the folder where the images are")
    parser.add_argument('-o', dest='output_path', default='epi') # choose the output folder
    parser.add_argument('--epi_h', default=False, action='store_true') # save the epi_h image
    parser.add_argument('--epi_v', default=False, action='store_true') # save the epi_v image
    parser.add_argument('--show', default=False, action='store_true') # show results!
    args = parser.parse_args()

    #pdb.set_trace()
    data_folder = args.data_folder[0]
    files_names = os.listdir(data_folder)
    images_names = [name for name in files_names if not name[0] == '.' and (name[-4:] == ".png" or name[-4:] == ".tif")]
    num_of_images = len(images_names)
    if num_of_images < 1:
        raise Exception('We did not found the images!\nSure they are in {}'.format(args.data_folder))

    num_of_views = np.sqrt(num_of_images).astype(int)
    if np.abs(num_of_views - np.sqrt(num_of_images)):
        message = 'The image number is weird, it should be a perfect square! How many views we have?\nNow we support only set of NxN subaperture views, sorry.'
        raise Exception(message)

    # sort so that they are in the order we want
    images_names.sort()

    # check out the sizes
    just_one_image_as_a_test = plt.imread(os.path.join(data_folder, images_names[0]))
    h = just_one_image_as_a_test.shape[0]
    w = just_one_image_as_a_test.shape[1]
    huge_width = w * num_of_views
    huge_height = h * num_of_views
    channels = just_one_image_as_a_test.shape[2]
    huge_image = np.zeros((huge_height, huge_width, channels))

    # create the epi if we want them
    if args.epi_h:
        epi_h = np.zeros((huge_height, just_one_image_as_a_test.shape[1], channels))
    if args.epi_v:
        epi_v = np.zeros((just_one_image_as_a_test.shape[0], huge_width, channels))

    # loop through images
    print("colleting the images and creating the EPI..")
    for j in range(num_of_views): # vertical
        for k in range(num_of_views): # horizontal

            image_name = os.path.join(data_folder, images_names[j*num_of_views+k])
            print("j:{}, k:{}, we read image {}".format(j, k, image_name))
            si = plt.imread(image_name)

            if args.epi_h:
                epi_h[j:epi_h.shape[0]:num_of_views, :] = si
            if args.epi_v:
                epi_v[:, k:epi_v.shape[1]:num_of_views, :] = si

            huge_image[j:huge_height:num_of_views, k:huge_width:num_of_views, :] = si

    # done!
    if args.show:
        if args.epi_h and args.epi_v:
            plt.subplot(221)
            plt.imshow(just_one_image_as_a_test)
            plt.title("the first of the images as a reference")
            plt.subplot(222)
            plt.imshow(epi_h)
            plt.title("Horizontal EPI")
            plt.subplot(223)
            plt.imshow(epi_v)
            plt.title("Vertical EPI")
            plt.subplot(224)
            plt.imshow(huge_image)
            plt.title("Huge Image")
            plt.show()

        else:
            plt.subplot(121)
            plt.imshow(just_one_image_as_a_test)
            plt.title("the first of the images as a reference")
            plt.subplot(122)
            plt.imshow(huge_image)
            plt.title("Huge Image")
            plt.show()

    # check output path
    out_path = args.output_path
    if out_path[0] == "/": #absolute Path
        print("output will be saved in {}".format(args.output_path))
    else:
        out_path = os.path.join(os.getcwd(), out_path)
        print("output will be saved in {}".format(out_path))

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    print("Saving..")
    huge_img_path = os.path.join(out_path, "lf.bmp")
    plt.imsave(huge_img_path, huge_image)
    if args.epi_h:
        epi_h_path = os.path.join(out_path, "epi_h.bmp")
        plt.imsave(epi_h_path, epi_h)
    if args.epi_v:
        epi_v_path = os.path.join(out_path, "epi_v.bmp")
        plt.imsave(epi_v_path, epi_v)
