import cv2 as cv
from utils import wait, resize
from preprocess import *
from extractor import *

if __name__ == "__main__":
    input_image = cv.imread("../app/Images/plates/2/2.jpg")
    # input_image = cv.imread("test_images/10.jpg")

    if input_image is None:
        print("\nInput image is not valid !")
        quit(0)

    resized = resize(input_image, 500)
    # cv.imshow('resized', resized)
    # wait()

    gray = get_grayscale_image(resized)
    # cv.imshow('grayscale', gray)
    # wait()

    blurred = get_blurred_image(gray)
    # cv.imshow('blurred', gray)
    # wait()

    derivated = get_image_directive(blurred)
    # cv.imshow('derivated', derivated)
    # wait()

    clipped = clip_histogram(derivated)
    # cv.imshow('clipped', clipped)
    # wait()

    binary = get_binary_image(clipped, 'OTSU')
    # cv.imshow('binary', binary)
    # wait()

    morphed = get_morphed(binary)

    contours = get_contours(morphed)

    plates = extract_candidate_rectangles(resized.copy(), contours)

    if len(plates) == 0:
        print("\ncouldn't find any plate, try other images")
        quit(0)
