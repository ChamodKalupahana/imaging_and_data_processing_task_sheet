
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
    plt.imshow(image)
    plt.show()
    plt.savefig(r"Task 4 Images/basic_image.jpeg")

    return 

def bad_pixel_interpolation():
    """
    Task b)

    Set the bad pixels to nans and calculate the all of the new pixel brightness first using neighbourhood averaging
    and then set the bad pixels to the new brightness
    """
    # load mask and mask array
    mask = np.loadtxt(r"Task sheet files-20221008\IRcombination\Near_IR_images\badpixel.mask", dtype=int) - 1
    masked_image = np.ones([100, 100])
    
    # load mask and image data  
    image = fits.getdata(r"Task sheet files-20221008\IRcombination\Near_IR_images\image01.fits")

    # set bad pixels to np.nan
    #masked_image = np.where(mask==image, image, np.nan)
    for i in range(np.shape(mask)[0]):
        image[mask[i][1]][mask[i][0]] = 0
        #masked_image = np.where(image==mask, image, 0)
    plt.imshow(image)
    plt.show()

    return


#test_astropy()
#display_image()
bad_pixel_interpolation()

