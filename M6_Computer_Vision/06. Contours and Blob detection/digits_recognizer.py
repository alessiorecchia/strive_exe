# from PIL.Image import CONTAINER
import cv2

import numpy as np
import matplotlib.pyplot as plt
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Conv2d,  MaxPool2d, ReLU, Sequential, BatchNorm2d, Dropout, Module, Linear
from torch import optim
from torchvision import datasets, transforms
from torchvision.transforms.transforms import Grayscale

from model import Net

model = Net()


BOX_COLOR = (0, 255, 0)
img = cv2.imread(sys.argv[1])

resized = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LANCZOS4)

def preprocessing(orig_image: np.ndarray, dil_iter = 3):
    # print('starting preprocessing')
    # print(orig_image.shape[:2])
    
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
    box = cv2.dilate(box, dilate_kernel, iterations=dil_iter)

    # plt.figure(figsize = (10,10))
    # plt.imshow(box, cmap='gray')
    # plt.show()
    # print('preprocessing ended...')

    return box

def get_surrounding_box(img: np.ndarray):
    # print(img.shape[:2])

    orig = img.copy()
    # print(orig.shape[:2])
    box = preprocessing(orig)
    # print('starting finding contours')

    # plt.figure(figsize = (10,10))
    # plt.imshow(box, cmap='gray')
    # plt.show()
    
      
    box_contours, h = cv2.findContours(box, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # print('Contours found: ', len(box_contours))
    # print('Contours coord: ', box_contours[0])
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
    angle = np.arctan(m) * 180 / np.pi
    
    return angle

def get_center(box):
    return (box[0,0] + box[2, 0])//2 , (box[0,1] + box[2, 1])//2

def get_translation_matrix(img, box_center):
    # print(img.shape[0], img.shape[1], box_center[0], box_center[1])
    x_shift = img.shape[0]//2 - box_center[0]
    y_shift = img.shape[1]//2 - box_center[1]

    matr = np.float32([
        [1, 0, x_shift],
        [0, 1, y_shift]
    ])

    return matr

def get_rotation_scale(img, box_dim):
    return max(img.shape)/max(box_dim)

def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

def to_mnist(img):
    _, img = cv2.threshold(img, 150, 255, cv2.THRESH_TRUNC)
    img = cv2.bitwise_not(img)
    
    kernel = np.ones((11, 11), np.uint8)
    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iteraions=1)
    img = cv2.dilate(img, kernel, iterations=1)
    # _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    # kernel_er = np.ones((7, 7), np.uint8)
    # img = cv2.erode(img, kernel_er, iterations=1)
    img = cv2.medianBlur(img, 7)

    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_LANCZOS4)
    
    return img

def to_tensor(arr: np.ndarray):
    tens = torch.from_numpy(arr).float()
    # tens = F.normalize(tens)
    # tens = (tens - 0.5)/0.5
    tens = tens.reshape(1, 1 ,28, 28)
    return tens







surr_box = get_surrounding_box(resized)
rotation = get_angle(surr_box)
center = get_center(surr_box)
print(center)
transl = get_translation_matrix(resized, center)
digits = resized.copy()
digits2 = digits.copy()

# cv2.drawContours(digits2, [surr_box], 0, BOX_COLOR, 2)

# plt.figure(figsize = (10,10))
# plt.imshow(cv2.cvtColor(digits2, cv2.COLOR_BGR2RGB))
# plt.show()


rot_scale = get_rotation_scale(digits, get_box_dim(surr_box))
rot_mat = cv2.getRotationMatrix2D((digits.shape[0]//2, digits.shape[1]//2), rotation, rot_scale)

w = max(digits.shape[:2])
h = min(digits.shape[:2])

# print(w, h)

digits = cv2.warpAffine(digits, transl, (w, h))
digits = cv2.warpAffine(digits, rot_mat, (w, h))

boxes = preprocessing(digits)

digits_contours, h = cv2.findContours(boxes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# print(len(digits_contours))

# digits_contours = sorted(digits_contours, key=cv2.contourArea, reverse=True)
digits_contours, bounding_boxes = sort_contours(digits_contours)
# print(digits_contours)

# digits_contours = digits_contours[:4]


digits_boxes = digits.copy()

for i, cont in enumerate(digits_contours):
    x,y,w,h = cv2.boundingRect(cont)

    digit = digits_boxes[y:y+h, x:x+w]
    plt.figure(figsize = (10,10))
    plt.imshow(cv2.cvtColor(digit, cv2.COLOR_BGR2RGB))
    plt.show()

    digit = cv2.cvtColor(digit, cv2.COLOR_RGB2GRAY)
    digit = to_mnist(digit)
    plt.figure(figsize=(10, 10))
    plt.imshow(digit, cmap='gray')
    plt.show()


    digit = to_tensor(digit)

    model.eval
    with torch.no_grad():

        # model = model.cuda()
        # three_mnist = three_mnist.cuda()

        ps = model(digit)
        # pred = torch.exp(ps)
        pred = F.softmax(ps)
        print(np.argmax(pred).item())





    # cv2.rectangle(digits_boxes,(x,y),(x+w,y+h),(0,0,255),2)
    # cv2.drawContours(digits_boxes, cont, 0, BOX_COLOR, 5)

# x,y,w,h = cv2.boundingRect(digits_contours)









# plt.figure(figsize = (10,10))
# plt.imshow(cv2.cvtColor(digits_boxes, cv2.COLOR_BGR2RGB))
# plt.show()


