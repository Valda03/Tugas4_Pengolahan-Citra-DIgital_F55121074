import numpy as np
import cv2
from matplotlib import pyplot as plt

# Memuat gambar
img = cv2.imread('min.png', cv2.IMREAD_GRAYSCALE)

# Menghitung DFT dari gambar
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# Mendefinisikan filter parameters
rows, cols = img.shape
crow, ccol = rows//2, cols//2
radius = 30
order = 2

# Membuat Butterworth Lowpass Filter
mask = np.zeros((rows, cols, 2), np.float32)
for i in range(rows):
    for j in range(cols):
        dist = np.sqrt((i-crow)**2 + (j-ccol)**2)
        mask[i, j] = 1 / (1 + (dist/radius)**(2*order))

# Mengaplikasikan filter pada DFT
dft_shift = dft_shift * mask

# Menghitung inverse DFT
idft_shift = np.fft.ifftshift(dft_shift)
idft = cv2.idft(idft_shift)
img_back = cv2.magnitude(idft[:, :, 0], idft[:, :, 1])

# Menampilkan original image, filter, dan filtered image
plt.subplot(131), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(mask[:, :, 0], cmap='gray')
plt.title('Butterworth Lowpass Filter'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(img_back, cmap='gray')
plt.title('Filtered Image'), plt.xticks([]), plt.yticks([])

plt.show()