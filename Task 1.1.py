import numpy as np
import matplotlib.pyplot as plt


def load_grey_image():
    # read the image and divude the image into rgb values
    orginal_image = plt.imread(r'Images\Orginal image - gym.jpeg')
    r, g, b = orginal_image[:,:,0], orginal_image[:,:,1], orginal_image[:,:,2]

    # there are different ways to making the image greyscale, 
    # the square root of the rgb values squared didn't produce images
    # that looked correct so this method was used using 
    # https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
    #greyscale_image = np.sqrt(r**2 + g**2 + b**2)

    greyscale_image = 0.2989 * r + 0.5870 * g + 0.1140 * b
    greyscale_image = (greyscale_image / np.amax(greyscale_image)).astype(np.float64)

    return orginal_image, greyscale_image

def plot_org_vs_grey_image(plot_histrogram):

    orginal_image, greyscale_image = load_grey_image()

    fig, ax = plt.subplots(1, 3, figsize=(10, 6))
    alpha = 10
    edit_greyscale_image = np.log(1 + alpha * greyscale_image) / np.log(1 + alpha)

    ax[0].imshow(orginal_image)
    ax[0].set_xticks([])
    ax[0].axes.get_yaxis().set_visible(False)
    ax[0].set_xlabel('Orginal Image')
    
    ax[1].imshow(greyscale_image, cmap='gray')
    ax[1].set_xticks([])
    ax[1].axes.get_yaxis().set_visible(False)
    ax[1].set_xlabel('Greyscale Image')

    ax[2].imshow(edit_greyscale_image, cmap='gray')
    ax[2].axes.get_yaxis().set_visible(False)
    ax[2].set_xlabel('Brighted Greyscale Image')
    
    plt.savefig('Images/greyscale_image.jpeg')

    if plot_histrogram == True:
        fig_2, ax_2 = plt.subplots(2)

        hist, bin_edges = np.histogram(greyscale_image, bins=125, range=(0, 1))
        ax_2[0].bar(bin_edges[0:-1], hist)
        ax_2[0].ticklabel_format(style='plain')

        hist, bin_edges = np.histogram(edit_greyscale_image, bins=125, range=(0, 1))
        ax_2[1].bar(bin_edges[0:-1], hist)
        ax_2[1].ticklabel_format(style='plain')

        fig_2.savefig('Images/histrogram_image.jpeg')
    plt.show()


plot_org_vs_grey_image(plot_histrogram=True)