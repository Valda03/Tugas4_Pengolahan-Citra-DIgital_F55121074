import numpy as np
import cv2
from matplotlib import pyplot as plt

# Memuat gambar
img = cv2.imread('unsharp.png', cv2.IMREAD_GRAYSCALE)

# Menghitung Laplacian darai gambar
laplacian = cv2.Laplacian(img, cv2.CV_64F)

# Mendefinisikan parameter untuk Unsharp Masking filter
alpha = 0.2
beta = 1.0 - alpha
gamma = 1.0

# Membuat Unsharp Masking filter
filter = gamma * (alpha * laplacian + beta * img)

# Menghitung DFT dari filter
dft = cv2.dft(np.float32(filter), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# Mendefinisikan filter parameters
rows, cols = img.shape
crow, ccol = rows//2, cols//2
d0 = 30

# Membuat Gaussian Lowpass Filter
mask = np.ones((rows, cols, 2), np.float32)
for i in range(rows):
    for j in range(cols):
        dist = np.sqrt((i-crow)**2 + (j-ccol)**2)
        mask[i, j] = np.exp(-(dist**2) / (2*(d0**2)))

# Mengaplikasikan filter ke shifted DFT
dft_shift = dft_shift * mask

# Menghitung inverse DFT
idft_shift = np.fft.ifftshift(dft_shift)
idft = cv2.idft(idft_shift)
img_back = cv2.magnitude(idft[:, :, 0], idft[:, :, 1])

# Menampilkan  original image, Laplacian, filter, dan filtered image
plt.subplot(221), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(laplacian, cmap='gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(filter, cmap='gray')
plt.title('Unsharp Masking Filter'), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(img_back, cmap='gray')
plt.title('Filtered Image'), plt.xticks([]), plt.yticks([])

plt.show()