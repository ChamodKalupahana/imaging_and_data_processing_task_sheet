
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

def display_image(image_path):
    """ Task a) Acquire images
    """
    # don't use fits.ope
    #image = fits.open(r"Task sheet files-20221008\IRcombination\Near_IR_images\image01.fits")

    image = fits.getdata(image_path)
    
    # plt.imshow() uses the astropy_mpl_style to show the image
    #plt.title('Unprocessed IR Image')
    plt.xticks([])
    plt.yticks([])

    plt_image = plt.imshow(image)
    plt.colorbar(plt_image)
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

        # use median for unbiased data
        kernel_mean = np.nanmean(kernel)

        surround_image[row, column] = kernel_mean
    
    interpolated_surround_image = surround_image[1:width + 1,1:height+1]

    return interpolated_surround_image


def bad_pixel_interpolation(image_path, remove_cosmis_rays, threshold, save_image, show_image):
    """
    Task b)

    Set the bad pixels to nans and calculate the all of the new pixel brightness first using neighbourhood averaging
    and then set the bad pixels to the new brightness

    Args:
        image_path (str): image_path to interpolate
        remove_cosmis_rays (boolan): Set cosmic ray pixels to nan and add their coords to mask to be interpolated
        save_image (boolan): Save image in Task 4 Images
        show_image (boolan): Show interpolated image

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
        # cosmic ray pixels are typically greater than 10000 for 1st image
        # cosmic ray pixels have 6000 brightess for 25th image
        # cosmic ray pixels have 5130 brightess for 3rd image
        #cosmic_ray_coords = np.where(image > 5130) # can't use this because end images have brightness of around 5000
        cosmic_ray_coords = np.where(image > threshold)
        
        # make sure returned cosmic_ray_coords in same shape as mask
        image[cosmic_ray_coords[0], cosmic_ray_coords[1]] = np.nan
        
        # add on cosmic ray positions to end of mask
        cosmic_ray_coords = np.transpose(cosmic_ray_coords)
        cosmic_ray_coords = np.transpose(np.array([cosmic_ray_coords[:,-1], cosmic_ray_coords[:,0]]))
        num_of_cosmic_rays = np.shape(cosmic_ray_coords)
        num_of_mask = np.shape(mask)
        mask = np.reshape(np.append(mask, cosmic_ray_coords), (num_of_mask[0] + num_of_cosmic_rays[0], 2))

    image = interpolate(image, mask)
    
    print(image_path[-12:-1],  'done')
    
    if show_image == True:
        plt_image = plt.imshow(image)
        #plt.title('Interpolated IR Image')
        plt.xticks([])
        plt.yticks([])
        plt.colorbar(plt_image)
        plt.show()
        if save_image == True:
            plt.savefig(r"Task 4 Images/bad_pixel_image.jpeg")

    return image

def sky_subtraction(show_brightness_plot, median_subtraction, inspection_subtraction, show_single_image, image_index):
    """The brightness of the sky increases as the images index increases

    Args:
        show_plot (boolan): _description_
    """

    image_path = glob.glob(r"Task sheet files-20221008\IRcombination\Near_IR_images\image*")
    
    # num of images
    N = np.size(image_path) 

    total_image = np.zeros([100, 100, N])
    total_subtracted_image = np.zeros([100, 100, N])
    
    # position of bottom left object in iamge01.fits is [8, 64]
    # position of bottom left object in iamge02.fits is [1, 64]

    # position of middle object in iamge01.fits is [63, 47] from inspection
    # position of middle object in iamge02.fits is [56, 46]
    # position of middle object in iamge03.fits is [50, 46]
    # position of middle object in iamge04.fits is [43, 46]
    # position of middle object in iamge05.fits is [37, 46]
    # and so...

    total_image_median = np.array([])

    for i in range(0, N):
        temp_image_path = image_path[i]
        temp_image = bad_pixel_interpolation(image_path=temp_image_path, remove_cosmis_rays=True, threshold=8000, save_image=False, show_image=False)
        total_image[:,:,i] = temp_image
        if median_subtraction == True:
            total_image_median = np.append(total_image_median, np.nanmedian(total_image[:, :, i]))
            total_subtracted_image[:,:,i] = total_image[:,:, i] - total_image_median[i]

    if show_brightness_plot == True:
        if inspection_subtraction == True:
            object_position = np.array([50, 46])
            object_brightness = total_image[object_position[0], object_position[1]]
            plt.plot(np.arange(0, N, 1), object_brightness, 'b-')
        if median_subtraction == True:
            plt.plot(np.arange(1, N + 1), total_image_median, 'b-')
            plt.savefig(r"Task 4 Images/background sky brightness.jpeg")
            plt.xlabel('Image number')
            plt.ylabel('Average Pixel Brightness')
            #plt.xticks(np.arange(1, N + 1, 1))
        plt.show()
    

    if show_single_image == True:
        image_index = int(image_index - 1) # array 0 for image 1

        image = total_image[:,:,image_index]
        subtracted_image = np.abs(image - total_image_median[image_index])
        subtracted_image = subtracted_image / np.max(subtracted_image)

        plt_image = plt.imshow(subtracted_image)
        plt.xticks([])
        plt.yticks([])
        plt.colorbar(plt_image)
        plt.show()

    return total_subtracted_image


def offset_determination():
    # make a 5x5 subplot
    total_subtracted_image = sky_subtraction(show_brightness_plot=False, median_subtraction=True, inspection_subtraction=False, show_single_image=False,
    image_index=1)

    combined_image = np.mean(total_subtracted_image, axis=2)
    plt.imshow(combined_image)
    plt.show()
    return

# image format is e.g image01, image25
image_path = r"Task sheet files-20221008\IRcombination\Near_IR_images\image25.fits"

#test_astropy()
display_image(image_path=image_path)
#bad_pixel_interpolation(image_path=image_path, remove_cosmis_rays=True,threshold=9000, save_image=True, show_image=True)

#sky_subtraction(show_brightness_plot=False, median_subtraction=True, inspection_subtraction=False, show_single_image=True,
#image_index=1)

#offset_determination()


