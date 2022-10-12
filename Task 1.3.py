import numpy as np
import matplotlib.pyplot as plt

orginal_image = plt.imread(r'C:\Users\chamo\Documents\Physics\Projects\Imaging and Data Processing\Images\Orginal image.jpeg')
r, g, b = orginal_image[:,:,0], orginal_image[:,:,1], orginal_image[:,:,2]

greyscale_image = np.sqrt(r**2 + g**2 + b**2)
greyscale_image = (greyscale_image / np.amax(greyscale_image)).astype(np.float64)

edit_greyscale_image = np.where(greyscale_image > 0.9, greyscale_image, 0)

fig, ax = plt.subplots(2, figsize=(10, 6))

hist, bin_edges = np.histogram(edit_greyscale_image, bins=256, range=(0, 1))
ax[0].plot(bin_edges[0:-1], hist)

ax[1].imshow(edit_greyscale_image, cmap='gray')
ax[1].axes.get_xaxis().set_visible(False)
ax[1].axes.get_yaxis().set_visible(False)
plt.savefig('Images/greyscale_image.jpeg')
plt.show()