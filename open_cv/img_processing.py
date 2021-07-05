import cv2
import numpy as np


def remove_noise(img):
    return cv2.blur(img, (5, 5), 0)


def get_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def thresholding(img):
    return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


# dilation
def dilate(img):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(img, kernel, iterations=1)


# erosion
def erode(img):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(img, kernel, iterations=1)


# open - erosion followed by dilation
def opening(img):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)


# canny edge detection
def canny(img):
    return cv2.Canny(img, 100, 200)


# skew correction
def deskew(img):
    coords = np.column_stack(np.where(img > 0))
    angle = cv2.minAreaRect(coords)[1]
    if angle < -45:
        angle = (-90 + angle)
    else:
        angle = -angle
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def match_template(img, template):
    return cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

def img_show():
    img = cv2.imread('pictures/test1.jpg', 1)
    kernel = np.ones((5, 5), np.float32) / 25
    # dst = cv2.filter2D(img, -1, kernel)
    dst = cv2.edgePreservingFilter(img, kernel, 0, 0, 0)
    cv2.imshow('Original', img)
    cv2.imshow('Result', dst)

    cv2.waitKey(0)

    # 이미지 윈도우 삭제
    cv2.destroyAllWindows()
    # 이미지 다른 파일로 저장
