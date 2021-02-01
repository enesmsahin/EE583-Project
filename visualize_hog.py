from skimage.feature import hog
from skimage.io import imread
from skimage.transform import rescale
import matplotlib.pyplot as plt


izmir = imread("./IMG_20210110_231147.jpg", as_gray=True)
metu_blue = imread("./IMG_20210110_231140.jpg", as_gray=True)
metu_red = imread("./IMG_20210110_231135.jpg", as_gray=True)
# izmir = rescale(izmir, 1/3, mode='reflect')
# metu_blue = rescale(metu_blue, 1/3, mode='reflect')
# metu_red = rescale(metu_red, 1/3, mode='reflect')

izmir_hog, izmir_hog_img = hog(
    izmir, pixels_per_cell=(8,8), 
    cells_per_block=(2, 2), 
    orientations=9, 
    visualize=True, 
    block_norm='L2-Hys')

metu_blue_hog, metu_blue_hog_img = hog(
    metu_blue, pixels_per_cell=(8,8), 
    cells_per_block=(2, 2), 
    orientations=9, 
    visualize=True, 
    block_norm='L2-Hys')

metu_red_hog, metu_red_hog_img = hog(
    metu_red, pixels_per_cell=(8,8), 
    cells_per_block=(2, 2), 
    orientations=9, 
    visualize=True, 
    block_norm='L2-Hys')

fig, ax = plt.subplots(2,3)
fig.set_size_inches(8,6)
# remove ticks and their labels
[a.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False) 
    for a in ax.reshape(-1)]

ax[0][0].imshow(izmir, cmap='gray')
ax[0][0].set_title('Izmir')
ax[0][1].imshow(metu_blue, cmap='gray')
ax[0][1].set_title('Metu_blue')
ax[0][2].imshow(metu_red, cmap='gray')
ax[0][2].set_title('Metu_red')


ax[1][0].imshow(izmir_hog_img, cmap='gray')
ax[1][0].set_title('Izmir HOG')
ax[1][1].imshow(metu_blue_hog_img, cmap='gray')
ax[1][1].set_title('Metu_blue HOG')
ax[1][2].imshow(metu_red_hog_img, cmap='gray')
ax[1][2].set_title('Metu_red HOG')
plt.show()