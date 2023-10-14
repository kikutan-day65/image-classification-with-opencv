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


def main():

    ## add path to pdf to be extracted
    # pdf_path = ''
    # extract_pdf(pdf_path)

    # arr = []

    dir = 'text'
    for filename in os.listdir(dir):
        img_path = os.path.join(dir, filename)

        sat = image_saturation(img_path)
        # arr.append(sat)

    # for i in arr:
    #     print(i)


if __name__ == "__main__":
    main()