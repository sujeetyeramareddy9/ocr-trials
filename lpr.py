import cv2
import imutils
import numpy as np
import pytesseract as pt

img = cv2.imread('data/test2.jpg')
img = cv2.resize(img, (620, 480))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5,5), 0)

edged = cv2.Canny(gray, 30, 200)
cv2.imshow('edges', edged)

contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
screenCnt = None

for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    x,y,w,h = cv2.boundingRect(c)
    if len(approx) == 4 and w > h:
        screenCnt = approx
        break
if screenCnt is not None:
    cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 4)

    mask = np.zeros(gray.shape, np.uint8)
    n_img = cv2.drawContours(mask, [screenCnt], 0, 255, -1)
    n_img = cv2.bitwise_and(img, img, mask=mask)

    x, y = np.where(mask == 255)
    topx, topy = np.min(x), np.min(y)
    bottomx, bottomy = np.max(x), np.max(y)
    cropped = gray[topx:bottomx+1, topy:bottomy+1]

    filtered = cv2.adaptiveThreshold(cropped, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7, 10)

    kernel = np.ones((1, 1), np.uint8)
    openit = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    closeit = cv2.morphologyEx(openit, cv2.MORPH_CLOSE, kernel)

    filtered = cv2.GaussianBlur(closeit, (5, 5), 0)

    text = pt.image_to_string(cropped, config='--oem 3 --psm 7')
    print("LICENSE PLATE DETECTED: " + text)
else:
    print('LICENSE PLATE NOT DETECTED')

cv2.imshow('filtered', filtered)
cv2.waitKey(0)
