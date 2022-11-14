
from ipaddress import collapse_addresses
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import astropy_mpl_style

# for masking
import glob

def test_astropy():
    import os
    import sys
    print(sys.version)
    import numpy as np
    print(np.__version__)  
    import matplotlib.pyplot as plt
    import matplotlib
    print('matplotlib: {}'.format(matplotlib.__version__))
    import scipy
    print("scipy:",scipy.__version__)
    import astropy as ast # module doesn't work, use link below
    
    #import astropy._erfa as efra
    pass

def display_image():
    """ Task a) Acquire images
    """
    # don't use fits.ope
    #image = fits.open(r"Task sheet files-20221008\IRcombination\Near_IR_images\image01.fits")

    image = fits.getdata(r"Task sheet files-20221008\IRcombination\Near_IR_images\image01.fits")
    
    # plt.imshow() uses the astropy_mpl_style to show the image
    plt.title('Unprocessed IR Image')
    plt.xticks([])
    plt.yticks([])

    plt.imshow(image)
    plt.savefig(r"Task 4 Images/basic_image.jpeg")
    plt.show()

    return 

def interpolate(image, mask):
    """ interpolate function for Task 4)b

    Args:
        image (array): 2D input image
        mask (array): 2D array of bad pixel coords

    Returns:
        array: interpolated image with bad pixels nanmean-ed
    """

    #------------------------------------------------------------------------
    #------------------ Add nans around the image  -----------------
    #------------------------------------------------------------------------
    
    # add an array of nans around the image for easier averaging
    # create a empty array of 100x100 zeros with 1x1 border around x and y axis
    width = np.shape(image)[0]
    height = np.shape(image)[1]
    surround_image = np.zeros([width + 2, height + 2])

    # replace 1x1 border with np.nans
    surround_image[0,:] = np.nan
    surround_image[-1,:] = np.nan
    surround_image[:,0] = np.nan
    surround_image[:,-1] = np.nan

    # replace the 100x100 array inside the border with the image
    surround_image[1:width + 1,1:height+1] = image[:,:]


    # try to avoid using for loops
    for k in mask:
        #column = mask[k + 1][1]
        #row = mask[k + 1][0]

        # add 1 to array indices since surround_image has 1 extra y and x array
        column = k[0] + 1
        row = k[1] + 1

        kernel = np.array([[surround_image[row - 1][column - 1], surround_image[row - 1][column ], surround_image[row - 1][column + 1]],
        [surround_image[row][column - 1], surround_image[row][column], surround_image[row][column + 1]],
        [surround_image[row + 1][column - 1], surround_image[row + 1][column], surround_image[row + 1][column + 1]]])

        kernel_mean = np.nanmean(kernel)

        surround_image[row, column] = kernel_mean
    
    interpolated_surround_image = surround_image[1:width + 1,1:height+1]

    return interpolated_surround_image


def bad_pixel_interpolation(image_path, remove_cosmis_rays):
    """
    Task b)

    Set the bad pixels to nans and calculate the all of the new pixel brightness first using neighbourhood averaging
    and then set the bad pixels to the new brightness

    Args:
        image_path (str): image_path to interpolate
        remove_cosmis_rays (boolan): Set cosmic ray pixels to nan and add their coords to mask to be interpolated

    Returns:
        array: 2D interpolated image with bad pixel removed

    """
    # load mask and mask array
    mask = np.loadtxt(r"Task sheet files-20221008\IRcombination\Near_IR_images\badpixel.mask", dtype=int) - 1
    image = fits.getdata(image_path)

    # num. of bad pixels
    n = np.shape(mask)[0]
    
        
    # set all bad pixels in image to np.nan
    image[mask[:,1], mask[:, 0]] = np.nan

    if remove_cosmis_rays == True:
        # cosmic ray pixels are typically greater than 10000
        cosmic_ray_coords = np.where(image > 10000)
        
        # make sure returned cosmic_ray_coords in same shape as mask
        image[cosmic_ray_coords[0], cosmic_ray_coords[1]] = np.nan
        
        # add on cosmic ray positions to end of mask
        cosmic_ray_coords = np.transpose(cosmic_ray_coords)
        cosmic_ray_coords = np.transpose(np.array([cosmic_ray_coords[:,-1], cosmic_ray_coords[:,0]]))
        num_of_cosmic_rays = np.shape(cosmic_ray_coords)
        num_of_mask = np.shape(mask)
        mask = np.reshape(np.append(mask, cosmic_ray_coords), (num_of_mask[0] + num_of_cosmic_rays[0], 2))

    image = interpolate(image, mask)

    plt.imshow(image)
    plt.savefig(r"Task 4 Images/bad_pixel_image.jpeg")
    plt.show()

    return image

def sky_subtraction():
    image_path = r"Task sheet files-20221008\IRcombination\Near_IR_images\image01.fits"
    bad_pixel_interpolation(image_path=image_path)
    return


image_path = r"Task sheet files-20221008\IRcombination\Near_IR_images\image01.fits"

#test_astropy()
#display_image()
image = bad_pixel_interpolation(image_path=image_path, remove_cosmis_rays=True)



