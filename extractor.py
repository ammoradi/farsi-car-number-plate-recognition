import numpy as np
import cv2 as cv
import math
from utils import wait, resize, pad_image
from preprocess import *


def get_contours(image):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    dilated = cv.dilate(image, kernel)
    _, contours, _ = cv.findContours(dilated.copy(), mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)
    return contours


def ratio_check(area, width, height):
    ratio = float(width) / float(height)
    if ratio < 1:
        ratio = 1 / ratio

    aspect = 4.7272
    min_area = 15*aspect*15  # minimum area
    max_area = 125*aspect*125  # maximum area

    min_ratio = 3
    max_ratio = 8

    if (area < min_area or area > max_area) or (ratio < min_ratio or ratio > max_ratio):
        return False
    return True


def is_max_white(plate):
    avg = np.mean(plate)
    if avg >= 115:
        return True
    else:
        return False


def validate_contour(rect):
    (x, y), (width, height), rect_angle = rect

    if width > height:
        angle = -rect_angle
    else:
        angle = 90 + rect_angle

    if angle > 15:
        return False

    if height == 0 or width == 0:
        return False

    area = height*width

    if not ratio_check(area, width, height):
        return False
    else:
        return True


def extract_candidate_rectangles(image, contours):
    rectangles = []
    for i, cnt in enumerate(contours):
        min_rect = cv.minAreaRect(cnt)

        if validate_contour(min_rect):
            x, y, w, h = cv.boundingRect(cnt)
            plate_img = image[y:y+h, x:x+w]

            if is_max_white(plate_img):
                copy = image.copy()
                cv.rectangle(copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
                rectangles.append(plate_img)
                cv.imshow("candidates", copy)
                cv.waitKey(0)
    return rectangles


def extract_number_areas(image):
    resized = resize(image, width=None, height=32)
    gray = get_grayscale_image(resized)
    blurred = get_blurred_image(gray)
    binary = get_binary_image(blurred, 'ADAPTIVE')

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    binary = cv.dilate(binary, kernel)

    plate_width = binary.shape[1]
    number_width = 18
    steps_number = int(math.ceil(plate_width / number_width))

    binary[binary == 0] = 250
    binary[binary == 255] = 0
    binary[binary == 250] = 255

    cv.imshow('eroded', binary)
    wait()

    numbers = []
    for i in range(steps_number):
        rect = binary[0:32, math.floor(i * number_width): math.ceil((i * number_width) + number_width)]
        # rect.reshape(32, 18, 1)
        brightness = np.average(rect)
        if 20 < brightness < 200:
            rect = pad_image(rect, left=7, right=7)
            numbers.append(rect)
        # cv.imshow('num' + str(i), rect)
        # wait()

    return numbers
