import cv2 as cv
import numpy as np
import os
from pdf2image import convert_from_path


def extract_pdf(pdf_path):
    pdf = convert_from_path(pdf_path)

    if pdf is None:
        print('you might add wrong path')

    for i in range(len(pdf)):
        pdf[i].save('page_' + str(i + 1) + '.jpg', 'JPEG')
    print('Extracted successfully!')


def main():

    # add path to pdf to be extracted
    # pdf_path = ''
    # extract_pdf(pdf_path)


if __name__ == "__main__":
    main()