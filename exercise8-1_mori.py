import cv2 as cv
import numpy as np
import os
from PIL import Image


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

    if blue_pixels_present > 90 or red_pixels_present > 90:
        return True # diagram
    else:
        return False # NOT diagram


def get_coordinates(img_path):
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

    if img is None:
        print ('Error opening image!')
        return -1
    
    edges = cv.Canny(img, 50, 150)

    # returns (x0, y0, x1, y1)
    coordinates = cv.HoughLinesP(edges, 1, np.pi / 180, 250, None, 60, 5)

    return coordinates


def get_gradient(coordinates):

    # get x0 and y0 coordinates
    x0, y0 = coordinates[:, :, 0], coordinates[:, :, 1]

    # get x1 and y1 coordinates
    x1, y1 = coordinates[:, :, 2], coordinates[:, :, 3]

    # Check if the denominator (x1 - x0) is zero
    denominator_zero = (x1 - x0) == 0

    # Calculate gradient, but set the value to 0 when denominator is zero
    gradient = (y1 - y0) // (x1 - x0)
    gradient[denominator_zero] = 0

    return gradient


def count_tilted_lines(gradients):
    lines = np.count_nonzero(gradients)

    return lines


def classify(img_path, filename, saturation, contains, lines):
    img = cv.imread(img_path)

    if saturation > 2:
        cv.imwrite(f'result/image/{filename}', img)
        return 0
    
    if contains is True:
        cv.imwrite(f'result/diagram/{filename}', img)
    else:

        if lines < 3:
            cv.imwrite(f'result/text/{filename}', img)
        else:
            cv.imwrite(f'result/diagram/{filename}', img)


def process_contour(img, img_path, contour, filename):
    peri = cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, 0.01 * peri, True)
    x, y, w, h = cv.boundingRect(contour)
    x2, y2 = x + w, y + h

    cv.rectangle(img, (x, y), (x2, y2), (0, 0, 255), 2)

    im = Image.open(img_path)
    im_crop = im.crop((x, y, x2, y2))
    im_crop.save(f'cutout-image/extracted_{filename}', quality=95)


def cut_out_pic(img_path, filename):
    img = cv.imread(img_path)

    upper = np.array([250, 255, 250])
    lower = np.array([0, 10, 5])

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower, upper)

    contours, hierarchy = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv.contourArea)
    c_sort = sorted(contours, key=cv.contourArea)
    c2 = c_sort[-2]

    process_contour(img, img_path, c, filename)
    process_contour(img, img_path, c2, filename)


def main():
    dir = f'resource'
    for filename in os.listdir(dir):
        img_path = os.path.join(dir, filename)

        sat = image_saturation(img_path)
        contains = contain_color(img_path)

        coordinates = get_coordinates(img_path)
        gradients = get_gradient(coordinates)
        lines = count_tilted_lines(gradients)
        
        classify(img_path, filename, sat, contains, lines)


    dir = 'result/image'
    for filename in os.listdir(dir):
        img_path = os.path.join(dir, filename)

        cut_out_pic(img_path, filename)


if __name__ == "__main__":
    main()