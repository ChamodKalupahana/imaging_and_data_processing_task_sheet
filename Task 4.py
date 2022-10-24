
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
    #import astropy.io.fits.util as fits # module doesn't work, use link below

    pass

def acquire_image(): 
    image = fits.open(r"Task sheet files-20221008\IRcombination\Near_IR_images\image01.fits")
    #mask = fits.open(r"Task sheet files-20221008\IRcombination\Near_IR_images\badpixel.mask")
    plt.imshow(image)
    plt.show()
    pass

#test_astropy()
acquire_image()
