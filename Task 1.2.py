import numpy as np
import matplotlib.pyplot as plt
from loading_data import load_grey_image

def segment_image(plot_histrogram, image, threshold):

    orginal_image, greyscale_image = load_grey_image(image=image)

    N = 3
    fig, ax = plt.subplots(1, N, figsize=(10, 6))
    edit_greyscale_image = np.where(greyscale_image > threshold / 256, greyscale_image, 0)
    edit_greyscale_image = np.where(greyscale_image < threshold / 256, edit_greyscale_image, 1)

    ax[0].imshow(orginal_image)
    ax[1].imshow(greyscale_image, cmap='gray')
    ax[2].imshow(edit_greyscale_image, cmap='gray')
    
    # plotting infomation
    for i in range(0, N):
        ax[i].set_xticks([])
        ax[i].axes.get_yaxis().set_visible(False)

    # for producing plt figure
    #ax[0].set_xlabel('Orginal Image')
    #ax[1].set_xlabel('Greyscale Image')
    #ax[2].set_xlabel('Binary Image')

    # for producing report figure
    ax[0].set_xlabel('(a)')
    ax[1].set_xlabel('(b)')
    ax[2].set_xlabel('(c)')
    
    plt.savefig('Task 1 Images/binary_image.jpeg')

    if plot_histrogram == True:
        fig_2, ax_2 = plt.subplots(2, figsize=(8, 7))

        hist, bin_edges = np.histogram(greyscale_image * 256, bins=100, range=(0, 256))
        ax_2[0].bar(bin_edges[0:-1], hist, align='edge', width=20)

        hist, bin_edges = np.histogram(edit_greyscale_image * 256, bins=100, range=(0, 256))
        ax_2[1].bar(bin_edges[0:-1], hist, align='edge', width=20)

        # plotting infomation
        # for producing plt figure
        #ax_2[0].set_xlabel('Orginal image')
        #ax_2[1].set_xlabel('Thresholded Image')
        
        # for producing report figure 
        ax_2[1].ticklabel_format(style='plain')
        ax_2[0].ticklabel_format(style='plain')
        ax_2[0].set_xlabel('Brightness of pixels')
        ax_2[1].set_xlabel('Brightness of pixels')
        ax_2[0].set_ylabel('Pixel percentage')
        ax_2[1].set_ylabel('Pixel percentage')
        ax_2[0].set_title('(a)')
        ax_2[1].set_title('(b)')
        
        fig_2.suptitle('Histograms of greyscale pixel brightness')
        fig_2.savefig('Task 1 Images/histrogram_binary_image.jpeg')
    plt.show()


# possible samples images are:
# original, ghost, goosefair, goosefair bright, goosefair day, gym, lake

# for gym image, use threshold=60
#segment_image(plot_histrogram=True, image='gym', threshold=60)


