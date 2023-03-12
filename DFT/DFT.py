import numpy as np
import cv2
from matplotlib import pyplot as plt

# Memuat Gambar
img = cv2.imread('dft.png', cv2.IMREAD_GRAYSCALE)

# Menghitung DFT dari gambar
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# Menghitung magnitude spectrum
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

# Menjalankan High Pass Filtering
rows, cols = img.shape
crow, ccol = rows//2, cols//2
mask = np.zeros((rows, cols, 2), np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1
dft_shift = dft_shift * mask

# Menghitung inverse DFT
idft_shift = np.fft.ifftshift(dft_shift)
idft = cv2.idft(idft_shift)
img_back = cv2.magnitude(idft[:, :, 0], idft[:, :, 1])

# Menampilkan gambar original dan magnitude spectrum
plt.subplot(131), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(img_back, cmap='gray')
plt.title('High Pass Filtered Image'), plt.xticks([]), plt.yticks([])

plt.show()