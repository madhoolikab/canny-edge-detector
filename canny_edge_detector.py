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

def threshold(non_max_suppressed_image, t1, t2):
    strong = np.zeros_like(non_max_suppressed_image, dtype = np.uint8)
    weak = np.zeros_like(non_max_suppressed_image, dtype = np.uint8)
    strong[non_max_suppressed_image > t1] = 1
    weak[non_max_suppressed_image > t2] = 1
    
    return (strong, weak)
    
def hysterisis(strong, weak):
    final_edge_map = strong.copy()
    M, N = strong.shape
    for i in range(M):
        for j in range(N):
            if weak[i, j] == 1:
                try:
                    if ((strong[i+1, j-1] == 1) or (strong[i+1, j] == 1) or (strong[i+1, j+1] == 1)
                        or (strong[i, j-1] == 1) or (strong[i, j+1] == 1)
                        or (strong[i-1, j-1] == 1) or (strong[i-1, j] == 1) or (strong[i-1, j+1] == 1)):
                        final_edge_map[i, j] = 1
                except IndexError as e:
                    pass
    return final_edge_map

# Read image to apply algorithm
img = cv2.imread('Lanes.jpg', cv2.IMREAD_GRAYSCALE)

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

# non-maximal suppresion
non_max_suppressed_image = non_maximal_suppression(magnitude_grad, angle_grad)
non_max_suppressed_image = np.divide(non_max_suppressed_image, 255)

# double-thresholding
strong = np.zeros_like(non_max_suppressed_image, dtype = np.uint8)
weak = np.zeros_like(non_max_suppressed_image, dtype = np.uint8)
(strong, weak) = threshold(non_max_suppressed_image, t1, t2)

# edge-tracking by hysterisis
final_edge_map = np.zeros_like(non_max_suppressed_image)
final_edge_map = hysterisis(strong, weak)

# plot the results
plt.figure()
plt.subplot(121)
plt.imshow(magnitude_grad, cmap = 'gray')
plt.subplot(122)
plt.imshow(non_max_suppressed_image, cmap = 'gray')

plt.figure()
plt.subplot(121)
weak_img_plt = plt.imshow(weak, cmap = 'gray')
plt.subplot(122)
strong_img_plt = plt.imshow(strong, cmap = 'gray')

plt.figure()
plt.subplots_adjust(left=0.1,bottom=0.35)
axcolor = 'lightgoldenrodyellow'
t1_axes = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
t2_axes = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
t1_vals = Slider(t1_axes, 'strong', 0, 0.5, valinit = t1)
t2_vals = Slider(t2_axes, 'weak', 0, 0.5, valinit = t2, slidermax = t1_vals)
t1_vals.slidermin = t2_vals
plt.subplot(111)
final_edge_plot = plt.imshow(final_edge_map, cmap = 'gray')

def update(val):
    t1 = t1_vals.val
    t2 = t2_vals.val
    strong = np.zeros_like(non_max_suppressed_image, dtype = np.uint8)
    weak = np.zeros_like(non_max_suppressed_image, dtype = np.uint8)
    (strong, weak) = threshold(non_max_suppressed_image, t1, t2)
    strong_img_plt.set_data(strong)
    weak_img_plt.set_data(weak)
    final_edge_map = np.zeros_like(non_max_suppressed_image)
    final_edge_map = hysterisis(strong, weak)
    final_edge_plot.set_data(final_edge_map)
    
t1_vals.on_changed(update)
t2_vals.on_changed(update)