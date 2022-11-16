import numpy as np
import matplotlib.pyplot as plt
from loading_data import load_grey_image
from scipy import ndimage

def convolution(image):
    # Horizontal Prewitt operator
    I_x = np.array([[-1,0,1],
    [-1,0,1],
    [-1,0,1]])
    
    # Vertical Prewitt operator
    I_y = np.array([[1,1,1],
    [0,0,0],
    [-1,-1,-1]])

    orginal_image, greyscale_image = load_grey_image(image=image)

    greyscale_image_I_x = ndimage.convolve(greyscale_image, I_x)
    greyscale_image_I_y = ndimage.convolve(greyscale_image, I_y)

    greyscale_image_egde = greyscale_image_I_x + greyscale_image_I_y
    
    N = 3 # num. of subplots
    fig, ax = plt.subplots(1, N)

    ax[0].imshow(orginal_image)
    ax[1].imshow(greyscale_image ,cmap='gray')
    ax[2].imshow(greyscale_image_egde, cmap='gray')

    # plotting infomation
    for i in range(0, N):
        ax[i].set_xticks([])
        ax[i].axes.get_yaxis().set_visible(False)
    
    # for producing report figure
    ax[0].set_xlabel('(a)')
    ax[1].set_xlabel('(b)')
    ax[2].set_xlabel('(c)')
    
    fig.savefig('Task 1 Images/edge_detection_image.jpeg')
    plt.show()

    return

# possible samples images are:
# original, ghost, goosefair, goosefair bright, goosefair day, gym, lake
convolution(image='lake')