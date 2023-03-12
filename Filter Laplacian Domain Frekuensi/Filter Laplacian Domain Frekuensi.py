import cv2
import numpy as np
from matplotlib import pyplot as plt

# Memuat gambar grayscale
img = cv2.imread('laplacian.jfif', 0)

# Menjalankan Discrete Fourier Transform
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# Mendefinisikan Laplacian of Gaussian filter
rows, cols = img.shape
sigma = 2.0
gaussian = cv2.getGaussianKernel(cols, sigma)
gaussian_2d = np.outer(gaussian, gaussian.transpose())
log_filter = -1 * cv2.Laplacian(gaussian_2d, cv2.CV_64F)

# Mengubah ukuran kernel untuk mencocokkan dimensi gambar
log_filter = cv2.resize(log_filter, (cols, rows))

# Mengaplikasikan filter untuk shifted Fourier spectrum
filtered_spectrum = fshift * log_filter

# Menjalankan Inverse Discrete Fourier Transform
filtered_img = np.real(np.fft.ifft2(np.fft.ifftshift(filtered_spectrum)))

# Menampilkan original dan filtered image
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(filtered_img, cmap='gray')
plt.title('Laplacian Filtered Image'), plt.xticks([]), plt.yticks([])
plt.show()
