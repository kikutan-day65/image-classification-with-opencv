import cv2 as cv
import numpy as np
import os
from pdf2image import convert_from_path


def extract_pdf(pdf_path):
    pdf = convert_from_path(pdf_path)

    if pdf is None:
        print('you might add wrong path')

    for i in range(len(pdf)):
        pdf[i].save('test/page_' + str(i + 1) + '.jpg', 'JPEG')
    print('Extracted successfully!')


def image_saturation(img_path):
    img = cv.imread(img_path)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    saturation = hsv[:, :, 1].mean()

    return saturation


def contain_color(img_path):
    img = cv.imread(img_path)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])

    lower_red = np.array([0, 50, 50])
    upper_red = np.array([20, 255, 255])

    blue_mask = cv.inRange(hsv, lower_blue, upper_blue)
    red_mask = cv.inRange(hsv, lower_red, upper_red)

    blue_pixels_present = cv.countNonZero(blue_mask)
    red_pixels_present = cv.countNonZero(red_mask)

    if blue_pixels_present > 0 or red_pixels_present > 0:
        return True # diagram
    else:
        return False # NOT diagram


def classify_by_saturation(saturation):
    if saturation <= 4:
        print(False) # NOT image
    else:
        print(True) # image


def main():

    # # add path to pdf to be extracted
    # pdf_path = 'pdfs_image_classification_task/pdfs/3.pdf'
    # extract_pdf(pdf_path)

    # arr = []

    dir = 'dia'
    for filename in os.listdir(dir):
        img_path = os.path.join(dir, filename)

        # sat = image_saturation(img_path)
        # classify_by_saturation(sat)

        contains = contain_color(img_path)

    # for i in arr:
    #     print(i)


if __name__ == "__main__":
    main()