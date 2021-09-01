import cv2
from scipy.ndimage import gaussian_filter


# Read image to apply algorithm
img = cv2.imread('Lanes.jpg', cv2.IMREAD_GRAYSCALE)
smoothened_img = gaussian_filter(img, 3)
M, N = img.shape