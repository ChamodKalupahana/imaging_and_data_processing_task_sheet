import numpy as np
import matplotlib.pyplot as plt

def load_grey_image(image):
    # read the image and divude the image into rgb values
    if image == 'gym':
        orginal_image = plt.imread(r'Images\Orginal image - gym.jpeg')

    if image == 'goosefair':
        orginal_image = plt.imread(r'Images\Orginal image - goosefair.jpeg')

    if image == 'ghost':
        orginal_image = plt.imread(r'Images\Orginal image - ghost.jpeg')

    if image == 'ocean':
        orginal_image = plt.imread(r'Images\Orginal image - ocean.jpeg')

    if image == 'orginal':
        orginal_image = plt.imread(r'Images\Orginal image.jpeg')
    r, g, b = orginal_image[:,:,0], orginal_image[:,:,1], orginal_image[:,:,2]

    # there are different ways to making the image greyscale, 
    # the square root of the rgb values squared didn't produce images
    # that looked correct so this method was used using 
    # https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
    #greyscale_image = np.sqrt(r**2 + g**2 + b**2)

    greyscale_image = 0.2989 * r + 0.5870 * g + 0.1140 * b
    greyscale_image = (greyscale_image / np.amax(greyscale_image)).astype(np.float64)

    return orginal_image, greyscale_image
