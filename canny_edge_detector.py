import cv2
from scipy.ndimage import gaussian_filter

def non_maximal_suppression(img, angle_grad):
    M, N = img.shape
    non_max_suppressed_image = np.zeros_like(img, dtype = np.int16)
            
    for i in range(M):
        for j in range(N):
            try:
                #angle 0
                if (0 <= angle_grad[i,j] < 22.5) or (157.5 <= angle_grad[i,j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                #angle 45
                elif (22.5 <= angle_grad[i,j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                #angle 90
                elif (67.5 <= angle_grad[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                #angle 135
                elif (112.5 <= angle_grad[i,j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]
                
                if (img[i,j] >= q) and (img[i,j] >= r):
                    non_max_suppressed_image[i,j] = img[i,j]
                else:
                    non_max_suppressed_image[i, j] = 0
            
            except IndexError as e:
                pass
            
    return non_max_suppressed_image

# Read image to apply algorithm
img = cv2.imread('Lanes.jpg', cv2.IMREAD_GRAYSCALE)
M, N = img.shape

# Gaussian Smoothing to reduce noise
smoothened_img = gaussian_filter(img, 3)


sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1],
                    [0, 0 ,0],
                    [1, 2, 1]])

grad_x = signal.convolve2d(smoothened_img, sobel_x, mode = 'same', boundary = 'symm')
grad_y = signal.convolve2d(smoothened_img, sobel_y, mode = 'same', boundary = 'symm')

magnitude_grad = np.sqrt((grad_x * grad_x) + (grad_y*grad_y))
angle_grad = 180.0 * np.arctan2(grad_y, grad_x)/np.pi
angle_grad[angle_grad <= 0] += 180

non_max_suppressed_image = non_maximal_suppression(magnitude_grad, angle_grad)
non_max_suppressed_image = np.divide(non_max_suppressed_image, 255)
