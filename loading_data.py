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


def load_affine_data():
    x_trans = np.loadtxt(r"Task sheet files-20221008\Affine\Affine\x_trans.mat", dtype=np.float64)
    y_trans = np.loadtxt(r"Task sheet files-20221008\Affine\Affine\y_trans.mat", dtype=np.float64)
    z_trans = np.loadtxt(r"Task sheet files-20221008\Affine\Affine\z_trans.mat", dtype=np.float64)
    
    room_corners = np.loadtxt(r"Task sheet files-20221008\Affine\Affine\room_corners.mat", dtype=np.float64)
    head = np.loadtxt(r"Task sheet files-20221008\Affine\Affine\head.mat", dtype=np.float64)

    alpha = np.loadtxt(r"Task sheet files-20221008\Affine\Affine\alpha.mat", dtype=np.float64)
    phi = np.loadtxt(r"Task sheet files-20221008\Affine\Affine\phi.mat", dtype=np.float64)
    theta = np.loadtxt(r"Task sheet files-20221008\Affine\Affine\theta.mat", dtype=np.float64)
    
    return x_trans, y_trans, z_trans, room_corners, head, alpha, phi, theta