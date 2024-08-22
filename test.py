import cv2 as cv
from skimage import img_as_ubyte, filters
import skimage as ski
import matplotlib.pyplot as plt

image = cv.imread('/root/workplace/opencv/speckle/test1.jpg')
print("image shape : ", image.shape)
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
print("gray_image shape : ", gray_image.shape)

cv.imwrite('/root/workplace/opencv/speckle/original_image.jpg', image)
cv.imwrite('/root/workplace/opencv/speckle/gray_image.jpg', gray_image)

Y = 150
gray_image_slice = gray_image[Y, :]
print(gray_image_slice.shape)
fig, ax = plt.subplots()
ax.plot(gray_image_slice, color='red')
ax.set_ylim(255, 0)
ax.set_ylabel('L, the intensity of the pixel')
ax.set_xlabel('X')

plot_path = '/root/workplace/opencv/speckle/plot1.jpg'
plt.savefig(plot_path)
plt.close()

image_blur = ski.filters.gamma(gray_image, sigma=3)
cv.imwrite('/root/workplace/opencv/speckle/blurred_image.jpg', img_as_ubyte(image_blur))

image_blur_pixels_slice = image_blur[Y, :]
image_blur_pixels_slice = ski.img_as_ubyte(image_blur_pixels_slice)
fig, ax = plt.subplots()
ax.plot(image_blur_pixels_slice, 'red')
ax.set_ylim(255, 0)
ax.set_ylabel('L, the intensity of the pixel')
ax.set_xlabel('X')

plot_path = '/root/workplace/opencv/speckle/plot2.jpg'
plt.savefig(plot_path)
plt.close()