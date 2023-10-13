import cv2 as cv
import numpy as np
import os
import fitz  # PyMuPDF
from PIL import Image


def calc_average_saturation(img_path):
    image = cv.imread(img_path)

    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    saturation_channel = hsv_image[:, :, 1]

    average_saturation = np.mean(saturation_channel)

    return average_saturation


def crop_image(img_path):
    img = cv.imread(img_path)

    crop = img[50:, 30:]

    return crop


def find_contours(img_path):
    img = crop_image(img_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (3, 3), 0)

    edges = cv.Canny(gray, 500, 600, apertureSize=3)

    contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    return len(contours)


def hough_line_p(img_path):
    img = crop_image(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    if img is None:
        print ('Error opening image!')
        return -1

    dst = cv.Canny(img, 200, 400, None, 3)

    cdstP = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)

    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 150, None, 50, 5)

    if linesP is None:
        return 0

    return len(linesP)


def contain_color(img_path):
    img = crop_image(img_path)

    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])

    lower_red = np.array([0, 50, 50])  # 下限（Hue, Saturation, Value）
    upper_red = np.array([20, 255, 255])

    blue_mask = cv.inRange(hsv_img, lower_blue, upper_blue)
    red_mask = cv.inRange(hsv_img, lower_red, upper_red)

    blue_pixels_present = cv.countNonZero(blue_mask)
    red_pixels_present = cv.countNonZero(red_mask)

    if blue_pixels_present > 0 or red_pixels_present > 0:
        return True
    else:
        return False


def thresholding(img_path):
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

    threshold, thresh = cv.threshold(img, 200, 255, cv.THRESH_BINARY)

    width, height = thresh.shape
    total_pixels = width * height

    black_pixels = np.sum(thresh == 0)
    percentage = (black_pixels / total_pixels) * 100

    return percentage


def classify(img_path, filename, percentage, saturation, contains, lines, contours, folder_num):

    img = cv.imread(img_path)

    if percentage > 80:
        return 0

    if saturation > 3:
        cv.imwrite(f'result{folder_num}/image/' + filename, img)
        return 0

    if contains is True:
        if contours > 2600 or 180 < contours < 350:
            cv.imwrite(f'result{folder_num}/text/' + filename, img)
            return 0
        elif 1270 < contours < 1391 or 160 < contours < 170:
            cv.imwrite(f'result{folder_num}/text/' + filename, img)
            return 0
        else:
            cv.imwrite(f'result{folder_num}/diagram/' + filename, img)
            return 0
    
    if lines <= 38:
        cv.imwrite(f'result{folder_num}/text/' + filename, img)
    elif 40 <= lines <= 45:
        cv.imwrite(f'result{folder_num}/text/' + filename, img)
    elif lines >= 220:
        cv.imwrite(f'result{folder_num}/text/' + filename, img)
    else:
        if contours > 1300:
            cv.imwrite(f'result{folder_num}/text/' + filename, img)
        elif 180 <= contours <= 206:
            cv.imwrite(f'result{folder_num}/text/' + filename, img)
        else:
            cv.imwrite(f'result{folder_num}/diagram/' + filename, img)
     

def main():

    for filename in os.listdir(dir):
        img_path = os.path.join(dir, filename)







    # for folder_num in range(1, 23):

    #     dir = f'test{folder_num}'

    #     for filename in os.listdir(dir):
    #         img_path = os.path.join(dir, filename)

    #         percentage = thresholding(img_path)
    #         saturation = calc_average_saturation(img_path)
    #         contains = contain_color(img_path)
    #         lines = hough_line_p(img_path)
    #         contours = find_contours(img_path)

    #         classify(img_path, filename, percentage, saturation, contains, lines, contours, folder_num)


if __name__ == "__main__":
    main()
