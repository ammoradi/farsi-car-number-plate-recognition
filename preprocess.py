import numpy as np
import cv2 as cv


def get_grayscale_image(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


def get_blurred_image(image, method='GAUSSIAN', kernel=(5, 5)):
    if method == 'GAUSSIAN':
        return cv.GaussianBlur(image, kernel, 0)
    elif method == 'BILATERAL':
        return cv.bilateralFilter(image, 9, 50, 50)
    else:
        return None


def get_image_directive(image, vertical=1, horizontal=0, kernel_size=3):
    return cv.Sobel(image, cv.CV_8U, vertical, horizontal, ksize=kernel_size)


def clip_histogram(image):
    img = image.copy()
    mean = np.mean(img)
    img[img < mean] = 0
    return img


def get_binary_image(image, method='OTSU'):
    if method == 'ADAPTIVE':
        return cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 115, 1)
    elif method == 'OTSU':
        ret, thresholded = cv.threshold(image, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        return thresholded
    elif method == 'BINARY':
        ret, thresholded = cv.threshold(image, 0, 255, cv.THRESH_BINARY)
        return thresholded
    else:
        return None


def get_morphed(image):
    img = image.copy()

    element = cv.getStructuringElement(shape=cv.MORPH_RECT, ksize=(17, 3))
    morph_img_threshold = img.copy()
    cv.morphologyEx(src=img, op=cv.MORPH_CLOSE, kernel=element, dst=morph_img_threshold)
    # cv.imshow("close morphed", morph_img_threshold)
    # cv.waitKey(0)

    element = cv.getStructuringElement(shape=cv.MORPH_RECT, ksize=(2, 4))
    cv.morphologyEx(src=morph_img_threshold, op=cv.MORPH_OPEN, kernel=element, dst=morph_img_threshold)
    # cv.imshow("open morphed", morph_img_threshold)
    # cv.waitKey(0)

    element = cv.getStructuringElement(shape=cv.MORPH_RECT, ksize=(30, 5))
    cv.morphologyEx(src=morph_img_threshold, op=cv.MORPH_CLOSE, kernel=element, dst=morph_img_threshold)
    # cv.imshow("close1 morphed", morph_img_threshold)
    # cv.waitKey(0)

    element = cv.getStructuringElement(shape=cv.MORPH_RECT, ksize=(5, 5))
    cv.morphologyEx(src=morph_img_threshold, op=cv.MORPH_OPEN, kernel=element, dst=morph_img_threshold)
    # cv.imshow("open1 morphed", morph_img_threshold)
    # cv.waitKey(0)

    element = cv.getStructuringElement(shape=cv.MORPH_RECT, ksize=(50, 5))
    cv.morphologyEx(src=morph_img_threshold, op=cv.MORPH_CLOSE, kernel=element, dst=morph_img_threshold)
    # cv.imshow("close2 morphed", morph_img_threshold)
    # cv.waitKey(0)

    return morph_img_threshold
