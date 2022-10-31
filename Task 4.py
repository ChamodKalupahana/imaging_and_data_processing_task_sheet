
from ipaddress import collapse_addresses
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import astropy_mpl_style

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
    for k in mask:
        column = mask[k + 1][1]
        row = mask[k + 1][0]

        #image[column][row] = np.array([[image[column - 1][row - 1], image[column][row - 1], image[column][row - 1]], /
        #[], /
        #[]])

    return


def bad_pixel_interpolation(show_bad_pixels):
    """
    Task b)

    Set the bad pixels to nans and calculate the all of the new pixel brightness first using neighbourhood averaging
    and then set the bad pixels to the new brightness

    # use np.copy to copy a new array
    # np.mask to get rid of cosmic rays
    # use np.isnan to match array with nans
    """
    # load mask and mask array
    mask = np.loadtxt(r"Task sheet files-20221008\IRcombination\Near_IR_images\badpixel.mask", dtype=int) - 1
    image = fits.getdata(r"Task sheet files-20221008\IRcombination\Near_IR_images\image01.fits")

    # num. of bad pixels
    n = np.shape(mask)[0]
    
    if show_bad_pixels == True:
        for i in range(n):
            # set all bad pixels in image to 0
            image[mask[i][1]][mask[i][0]] = 0
        
    if show_bad_pixels == False:        
        for i in range(n):
            # set all bad pixels in image to np.nan
            image[mask[i][1]][mask[i][0]] = np.nan

    # add an array of nans around the image for easier averaging
    
    nan_array = np.empty(100)
    #nan_array[:] = np.nan
    nan_array[:] = 0
    image = np.insert(image, 0, nan_array, axis=0)
    image = np.insert(image, 101, nan_array, axis=0)

    interpolate()

    plt.imshow(image)
    plt.savefig(r"Task 4 Images/bad_pixel_image.jpeg")
    plt.show()


    return


#test_astropy()
#display_image()
bad_pixel_interpolation(show_bad_pixels=True)

