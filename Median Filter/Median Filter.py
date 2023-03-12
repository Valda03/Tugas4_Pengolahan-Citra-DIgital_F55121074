import cv2

# memuat gambar
img = cv2.imread('median.jpg', cv2.IMREAD_GRAYSCALE)

# Mendefiniskan ukuran kernel untuk median filter
kernel_size = 3

# mengaplikasikan median filter
filtered_img = cv2.medianBlur(img, kernel_size)

# Menampilkan gambar original dan gambar filter
cv2.imshow('Gambar Original', img)
cv2.imshow('Gambar Filter', filtered_img)
cv2.waitKey(0)
cv2.destroyAllWindows()