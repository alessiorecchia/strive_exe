# from PIL.Image import CONTAINER
import cv2

import numpy as np
import matplotlib.pyplot as plt
import sys

BOX_COLOR = (0, 255, 0)
img = cv2.imread(sys.argv[1])

resized = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LANCZOS4)

def preprocessing(orig_image: np.ndarray):
    # print('starting preprocessing')
    print(orig_image.shape[:2])
    
    kernel_dim = int(max(orig_image.shape) * 0.02525) if int(max(orig_image.shape) * 0.02525) % 2 != 0 else int(max(orig_image.shape) * 0.02525) + 1
    sd = kernel_dim * 0.0495
    
    box = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
    gauss_kernel = (kernel_dim, kernel_dim)
    box = cv2.GaussianBlur(box, gauss_kernel, sd)

    # plt.figure(figsize = (10,10))
    # plt.imshow(box, cmap='gray')
    # plt.show()

    _, box = cv2.threshold(box, 128, 255, cv2.THRESH_BINARY_INV)

    # plt.figure(figsize = (10,10))
    # plt.imshow(box, cmap='gray')
    # plt.show()
    
    dilate_kernel = np.ones((gauss_kernel[0]//2, gauss_kernel[1]//2), np.uint8)
    box = cv2.dilate(box, dilate_kernel, iterations=4)

    # plt.figure(figsize = (10,10))
    # plt.imshow(box, cmap='gray')
    # plt.show()
    # print('preprocessing ended...')

    return box

def get_surrounding_box(img: np.ndarray):
    print(img.shape[:2])

    orig = img.copy()
    print(orig.shape[:2])
    box = preprocessing(orig)
    # print('starting finding contours')

    # plt.figure(figsize = (10,10))
    # plt.imshow(box, cmap='gray')
    # plt.show()
    
      
    box_contours, h = cv2.findContours(box, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print('Contours found: ', len(box_contours))
    print('Contours coord: ', box_contours[0])
    cnt = box_contours[0]
    # cv2.drawContours(orig, [cnt], 0, BOX_COLOR, 2)

    # plt.figure(figsize = (10,10))
    # plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
    # plt.show()

    minRect = cv2.minAreaRect(box_contours[0])

    box = cv2.boxPoints(minRect)
    box = np.intp(box)
    # print('finding contours ended...')

    # print(box)
    
    return box


def get_box_dim(box):
    assert box.shape[0] == 4
    dim = []
    for i in range(box.shape[0] - 2):
        dim.append(np.sqrt((box[i, 0] - box[i+1, 0])**2 + (box[i, 1] - box[i+1, 1])**2))
    
    return np.array(dim)


def get_angle(box):
    
    dist = get_box_dim(box)

    start = dist.argmax()

    m = (surr_box[start+1, 1] - surr_box[start, 1]) / (surr_box[start+1, 0] - surr_box[start, 0])
    dir = m/abs(m)

    angle = np.arctan(m) * 180 / np.pi
    
    return angle

def get_center(box):
    return (box[0,0] + box[2, 0])//2 , (box[0,1] + box[2, 1])//2

def get_translation_matrix(img, box_center):
    print(img.shape[0], img.shape[1], box_center[0], box_center[1])

    x_shift = img.shape[0]//2 - box_center[0]
    y_shift = img.shape[1]//2 - box_center[1]

    matr = np.float32([
        [1, 0, x_shift],
        [0, 1, y_shift]
    ])

    return matr

def get_rotation_scale(img, box_dim):
    return max(img.shape)/max(box_dim)






surr_box = get_surrounding_box(resized)
rotation = get_angle(surr_box)
center = get_center(surr_box)

transl = get_translation_matrix(resized, center)

digits = resized.copy()
digits2 = digits.copy()

cv2.drawContours(digits2, [surr_box], 0, BOX_COLOR, 2)

plt.figure(figsize = (10,10))
plt.imshow(cv2.cvtColor(digits2, cv2.COLOR_BGR2RGB))
plt.show()


rot_scale = get_rotation_scale(digits, get_box_dim(surr_box))
rot_mat = cv2.getRotationMatrix2D((digits.shape[0]//2, digits.shape[1]//2), rotation, rot_scale)

w = max(digits.shape[:2])
h = min(digits.shape[:2])

print(w, h)

digits = cv2.warpAffine(digits, transl, (w, h))
digits = cv2.warpAffine(digits, rot_mat, (w, h))



## finding digits boundaries
boxes = digits.copy()
kernel_dim = int(max(digits.shape) * 0.02525) if int(max(digits.shape) * 0.02525) % 2 != 0 else int(max(digits.shape) * 0.02525) + 1
st = kernel_dim * 0.0495

boxes = cv2.cvtColor(boxes, cv2.COLOR_BGR2GRAY)
gauss_kernel = (kernel_dim, kernel_dim)
boxes = cv2.GaussianBlur(boxes, gauss_kernel, st)
# boxes = cv2.adaptiveThreshold(boxes, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 3)



_, boxes = cv2.threshold(boxes, 128, 255, cv2.THRESH_BINARY_INV)

plt.figure(figsize = (10,10))
plt.imshow(boxes, cmap='gray')
plt.show()

dilate_kernel = np.ones((gauss_kernel[0]//2, gauss_kernel[1]//2), np.uint8)
boxes = cv2.dilate(boxes, dilate_kernel, iterations=2)

plt.figure(figsize = (10,10))
plt.imshow(boxes, cmap='gray')
plt.show()

# boxes = cv2.Canny(boxes, 100, 200)

# plt.figure(figsize = (10,10))
# plt.imshow(boxes, cmap='gray')
# plt.show()

digits_contours, h = cv2.findContours(boxes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

print(len(digits_contours))

digits_contours = sorted(digits_contours, key=cv2.contourArea, reverse=True)
# print(digits_contours)

# digits_contours = digits_contours[:4]


digits_boxes = digits.copy()

for i, cont in enumerate(digits_contours):
    x,y,w,h = cv2.boundingRect(cont)
    cv2.rectangle(digits_boxes,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.drawContours(digits_boxes, cont, 0, BOX_COLOR, 5)

# x,y,w,h = cv2.boundingRect(digits_contours)









plt.figure(figsize = (10,10))
plt.imshow(cv2.cvtColor(digits_boxes, cv2.COLOR_BGR2RGB))
plt.show()


