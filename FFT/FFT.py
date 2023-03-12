import numpy as np
import cv2
from matplotlib import pyplot as plt

# Memuat Gambar
img = cv2.imread('fft.jfif', cv2.IMREAD_GRAYSCALE)

# Menjalankan Fourier Transform
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

# Menjalankan High Pass Filtering
rows, cols = img.shape
crow, ccol = rows//2, cols//2
fshift[crow-30:crow+30, ccol-30:ccol+30] = 0
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

# Menampilkan gambar original dan magnitude spectrum
plt.subplot(131), plt.imshow(img, cmap='gray')
plt.title('Gambar original'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(img_back, cmap='gray')
plt.title('Gambar High Pass Filter'), plt.xticks([]), plt.yticks([])

plt.show()
