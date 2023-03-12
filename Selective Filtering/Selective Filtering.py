import cv2
import numpy as np
from matplotlib import pyplot as plt

# Memuat gambar grayscale
img = cv2.imread('xray.jpg', 0)

# Menghitung 2D Fourier Transform dari gambar
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# Mendefiniskan selective filter
rows, cols = img.shape
crow, ccol = int(rows/2), int(cols/2)
radius = 30
mask = np.ones((rows, cols), np.uint8)
cv2.circle(mask, (ccol, crow), radius, 0, -1)

# Mengaplikasikan selective filter ke Fourier Transform dari image
filtered_spectrum = fshift * mask

# Mengganti filtered spectrum kembali ke lokasi original
f_ishift = np.fft.ifftshift(filtered_spectrum)

# Menjalankan inverse Fourier Transform untuk mendapatkan gambar kembali ke domain spasial
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

# Menampilkan gambar original and filtered
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
plt.title('Filtered Image'), plt.xticks([]), plt.yticks([])
plt.show()
