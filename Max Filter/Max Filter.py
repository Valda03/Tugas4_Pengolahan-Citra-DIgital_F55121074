import numpy as np
import cv2

# Memuat Gambar
img = cv2.imread('min.png', cv2.IMREAD_GRAYSCALE)

# Mendefinisikan ukuran kernel untuk max filter
kernel_size = 3

# Pengaplikasian max filter
kernel = np.ones((kernel_size, kernel_size), np.uint8)
filtered_img = cv2.dilate(img, kernel, iterations = 1)

# Menampilkan gambar original dan gambar filter
cv2.imshow('Gambar Original', img)
cv2.imshow('Gambar Filter', filtered_img)
cv2.waitKey(0)
cv2.destroyAllWindows()