import cv2 as cv


def wait():
    cv.waitKey(0)


def resize(image, width, height):
    if height is None:
        (h, w) = image.shape[:2]
        r = width / float(w)
        dim = (width, int(h * r))
        return cv.resize(image, dim, interpolation=cv.INTER_AREA)

    elif width is None:
        (h, w) = image.shape[:2]
        r = height / float(h)
        dim = (int(w * r), height)
        return cv.resize(image, dim, interpolation=cv.INTER_AREA)

    else:
        dim = (width, height)
        return cv.resize(image, dim, interpolation=cv.INTER_AREA)


def pad_image(image, top=0, right=0, bottom=0, left=0):
    img = image.copy()
    bordered = cv.copyMakeBorder(img, top=top, bottom=bottom, left=left, right=right,
               borderType=cv.BORDER_CONSTANT, value=[0, 0, 0])
    return bordered
