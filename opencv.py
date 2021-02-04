import numpy as np
import cv2
import ffmpeg

import pytesseract as pt
import matplotlib.pyplot as plt

# ###############################
# Drawing shapes and writing text
# ###############################

img = np.zeros((512, 512, 3), np.uint8)

img = cv2.line(img, (0, 0), (262, 512), (0, 0, 255), 5)

img = cv2.rectangle(img, (262, 0), (512, 262), (0, 255, 0), 3)

img = cv2.circle(img, (384, 384), 50, (255, 0, 0), -1)

img = cv2.ellipse(img, (262, 262), (100, 50), 0, 0, 360, (255, 255, 0), 3)

poly_pts = np.array([[128, 128], [178, 128], [153, 178]]).reshape((-1, 1, 2))
img = cv2.polylines(img, [poly_pts], False, (0, 255, 255), 2)

font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
cv2.putText(img, 'Testing', (10, 500), font, 4, (255, 255, 255), 2)

cv2.imwrite('test.jpg', img)

# ##########################
# Basic Operations on Images
# ##########################

img = cv2.imread('bron.jpg')

print(img[100, 100])
print(img.item(100, 100, 1))  # accessing green value
print(img.shape)
print(img.size)

# ###############################
# Arithmetic Operations on Images
# ###############################

img1 = cv2.imread('bron.jpg')
img2 = cv2.imread('lal.jpg')

rows, cols, channels = img2.shape
roi = img1[0:rows, 0:cols]

img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 90, 255, cv2.THRESH_BINARY)

mask = cv2.bitwise_not(mask)
mask_inv = cv2.bitwise_not(mask)

img1bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
img2fg = cv2.bitwise_and(img2, img2, mask=mask)

dst = cv2.add(img1bg, img2fg)
img1[0:rows, 0:cols] = dst


# cv2.imshow('LBJ2LAL', img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ##########################################################
# Changing Colorspaces & Object Tracking Using ffmpeg-python
# ##########################################################

def check_rotation(path):
    meta_data = ffmpeg.probe(path)

    rCode = None
    if int(meta_data['streams'][0]['tags']['rotate']) == 90:
        rCode = cv2.ROTATE_90_CLOCKWISE
    elif int(meta_data['streams'][0]['tags']['rotate']) == 180:
        rCode = cv2.ROTATE_180
    elif int(meta_data['streams'][0]['tags']['rotate']) == 270:
        rCode = cv2.ROTATE_90_COUNTERCLOCKWISE

    return rCode


def correct_rotation(f, rCode):
    return cv2.rotate(f, rCode)


cap = cv2.VideoCapture('0828_last.mov')
rotateCode = check_rotation('0828_last.mov')

size = cap.read()[1].shape
scale = 0.6
dim = (int(size[0] * scale), int(size[1] * scale))

key = cv2.waitKey(1)

# print('hsv of blue:')
# blue = np.uint8([[[255, 255, 255]]])
# print(cv2.cvtColor(blue, cv2.COLOR_BGR2HLS))

kernel = np.ones((7, 7), np.uint8)

while True:
    key = cv2.waitKey(1)
    if key == ord('p'):
        cv2.waitKey(-1)

    _, frame = cap.read()

    if not _:
        break

    if rotateCode is not None:
        frame = correct_rotation(frame, rotateCode)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, dim)
    frame = cv2.resize(frame, dim)
    frame = cv2.GaussianBlur(frame, (5,5), 0)
    #frame = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=-1)
    #frame = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=-1)

    #frame = cv2.Canny(frame, 245, 254)
    #frame = cv2.morphologyEx(frame, cv2.MORPH_TOPHAT, kernel)

    cv2.imshow('frame', frame)

    # lower = 190
    # upper = 210

    # ret1, thresh1 = cv2.threshold(gray, lower, upper, cv2.THRESH_BINARY)
    # ret2, thresh2 = cv2.threshold(gray, lower, upper, cv2.THRESH_BINARY_INV)
    # ret3, thresh3 = cv2.threshold(gray, lower, upper, cv2.THRESH_TRUNC)
    # ret4, thresh4 = cv2.threshold(gray, lower, upper, cv2.THRESH_TOZERO)
    # ret5, thresh5 = cv2.threshold(gray, lower, upper, cv2.THRESH_TOZERO_INV)
    #
    # thresh6 = cv2.adaptiveThreshold(gray, 100, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 7)

    # mask = cv2.inRange(gray, lower_white, upper_white)
    # result = cv2.bitwise_and(frame, frame, mask=mask)

    # cv2.imshow('thresh1', thresh1)
    # cv2.imshow('thresh2', thresh2)
    # cv2.imshow('thresh3', thresh3)
    # cv2.imshow('thresh4', thresh4)
    # cv2.imshow('thresh6', thresh6)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
