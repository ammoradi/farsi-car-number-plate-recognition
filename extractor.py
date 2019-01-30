import numpy as np
import cv2 as cv


def get_contours(image):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    dilated = cv.dilate(image, kernel)
    # cv.imshow('dilated', dilated)
    # cv.waitKey(0)
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
                cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                rectangles.append((x, y, w, h))
                cv.imshow("candidates", image)
                cv.waitKey(0)
    return rectangles
