import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt


def calc_average_saturation(img_path):
    image = cv.imread(img_path)

    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    saturation_channel = hsv_image[:, :, 1]

    average_saturation = np.mean(saturation_channel)

    return average_saturation


def hough_lines(img_path):
    img = cv.imread(img_path)
    img = cv.resize(img, (600, 900))
    img = cv.GaussianBlur(img, (5, 5), 1)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    edges = cv.Canny(gray, 150, 300, apertureSize=5)

    # thresh_val = 130
    # _, bin_img = cv.threshold(gray, thresh_val, 255, cv.THRESH_BINARY)

    lines = cv.HoughLines(edges, 1, np.pi/180, 250)

    if lines is None:
        return 0

    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)

        x0  = a * rho
        y0 = b * rho

        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # cv.imshow('houghlines', img)

    cv.waitKey(0)
    cv.destroyAllWindows()

    return len(lines)


def find_contours(img_path):
    img = cv.imread(img_path)
    img = cv.resize(img, (600, 900))
    img = cv.GaussianBlur(img, (5, 5), 1)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    edges = cv.Canny(gray, 400, 500, apertureSize=5)

    contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    cv.drawContours(img, contours, -1, (255, 0, 0), 1)

    cv.imshow('contours', img)

    cv.waitKey(0)
    cv.destroyAllWindows()


def classify_image_and_others(img_path, filename, saturation):
    img = cv.imread(img_path)

    if saturation > 3:
        cv.imwrite('result/image/' + filename, img)
    else:
        cv.imwrite('result/text_and_diagram/' + filename, img)


def thresholding(img_path):
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    img = cv.resize(img, (600, 900))
    img = cv.GaussianBlur(img, (5, 5), 1)

    _, thresh = cv.threshold(img, 250, 255, cv.THRESH_BINARY)

    canny = cv.Canny(thresh, 150, 250, apertureSize=3)
    
    cv.imshow('bin', canny)

    cv.waitKey(0)
    cv.destroyAllWindows()


def main():

    img_path = 'dia/test-244.jpg'

    thresholding(img_path)
    # corners = harris_corners(img_path)

    # percentage = thresholding(img_path)
    # print(percentage)

    # lines = hough_lines(img_path)
    # print("lines: ", lines)

    # cont = find_contours(img_path)
    # print("contours: ", cont)

    # percentage = thresholding(img_path)
    # print(f'black pix: {percentage} %')

    # dir = 'dia'
    # for filename in os.listdir(dir):
    #     img_path = os.path.join(dir, filename)

    #     matching(img_path, filename)

if __name__ == "__main__":
    main()
