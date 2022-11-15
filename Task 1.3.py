import numpy as np
import matplotlib.pyplot as plt
from loading_data import load_grey_image
from scipy import ndimage

def convolution(image):
    Gx = np.array([[-1,0,1],
    [-1,0,1],
    [-1,0,1]])

    Gy = np.array([[1,1,1],
    [0,0,0],
    [-1,-1,-1]])

    orginal_image, greyscale_image = load_grey_image(image=image)

    Ex = ndimage.convolve(greyscale_image,Gx)
    Ey = ndimage.convolve(greyscale_image, Gy)

    Edgeimage = Ex + Ey

    plt.imshow(Edgeimage,cmap='gray')
    plt.show()
    return

convolution(image='ghost')