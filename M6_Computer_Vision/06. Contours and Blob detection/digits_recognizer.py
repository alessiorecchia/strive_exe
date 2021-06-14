import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

BOX_COLOR = (0, 255, 0)

img = cv2.imread(sys.argv[1])

box = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gauss_kernel = (35, 35)
box = cv2.GaussianBlur(box, gauss_kernel, 2)
box = cv2.adaptiveThreshold(box, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 5)
box = cv2.bitwise_not(box)

plt.figure(figsize = (10,10))
plt.imshow(box, cmap='gray')
plt.show()

_, box = cv2.threshold(box, 128, 255, cv2.THRESH_BINARY)
dilate_kernel = np.ones((gauss_kernel[0]//2, gauss_kernel[1]//2), np.uint8)
box = cv2.dilate(box, dilate_kernel, iterations=10)

plt.figure(figsize = (10,10))
plt.imshow(box, cmap='gray')
plt.show()

box_contours, h = cv2.findContours(box, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)



accuracy = 0.3

digits = img.copy()

for contour in box_contours:
    perimeter = cv2.arcLength(contour, True)
    eps = perimeter * accuracy
    approx = cv2.approxPolyDP(contour, eps, True)
    cv2.drawContours(digits, [approx], 0, BOX_COLOR, 2)

print(approx)






plt.figure(figsize = (10,10))
plt.imshow(digits, cmap='gray')
plt.show()


