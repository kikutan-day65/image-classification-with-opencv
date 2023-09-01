import cv2 as cv
import numpy as np
import os


def calc_average_saturation(img_path):
    image = cv.imread(img_path)

    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    saturation_channel = hsv_image[:, :, 1]

    average_saturation = np.mean(saturation_channel)

    return average_saturation


def classify_image_and_others(img_path, filename, saturation):
    img = cv.imread(img_path)

    if saturation > 3:
        cv.imwrite('result/image/' + filename, img)
    else:
        cv.imwrite('result/text_and_diagram/' + filename, img)



def main():

    img_path = 'test/test-046.jpg'


if __name__ == "__main__":
    main()
