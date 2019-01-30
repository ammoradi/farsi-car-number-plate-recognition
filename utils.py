import cv2 as cv


def wait():
    cv.waitKey(0)


def resize(image, width):
    (h, w) = image.shape[:2]
    r = width / float(w)
    dim = (width, int(h * r))
    return cv.resize(image, dim, interpolation=cv.INTER_AREA)
