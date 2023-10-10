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


def classify_image_and_others(img_path, filename, saturation):
    img = cv.imread(img_path)

    if saturation > 3:
        cv.imwrite('result/image/' + filename, img)
    else:
        cv.imwrite('result/text_and_diagram/' + filename, img)


def crop_image(img_path):
    img = cv.imread(img_path)

    crop = img[50:, 30:]

    # cv.imshow('cropped', crop)

    # cv.waitKey(0)
    # cv.destroyAllWindows()

    return crop


def find_contours(img_path):
    img = crop_image(img_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (3, 3), 0)

    edges = cv.Canny(gray, 500, 600, apertureSize=3)

    contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    cv.drawContours(img, contours, -1, (255, 0, 0), 2)

    # cv.imshow('contours', img)

    # cv.waitKey(0)
    # cv.destroyAllWindows()

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
    
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
    
    # cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
    
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    return len(linesP)


def contain_color(img_path):
    img = crop_image(img_path)

    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])

    blue_mask = cv.inRange(hsv_img, lower_blue, upper_blue)

    blue_pixels_present = cv.countNonZero(blue_mask)

    if blue_pixels_present > 0:
        return True
    else:
        return False


def thresholding(img_path):
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

    threshold, thresh = cv.threshold(img, 200, 255, cv.THRESH_BINARY)

    # ピクセル数を取得
    width, height = thresh.shape
    total_pixels = width * height

    black_pixels = np.sum(thresh == 0)

    # print("total_pixels:", total_pixels)
    # print('Number of black pixels:', black_pixels)

    # 黒ピクセルの割合
    percentage = (black_pixels / total_pixels) * 100

    # cv.imshow('bin', thresh)

    cv.waitKey(0)
    cv.destroyAllWindows()

    return percentage


def classify_by_color(img_path, filename, contains):
    img = cv.imread(img_path)

    if contains is True:
        cv.imwrite('result/diagram/' + filename, img)
    else:
        cv.imwrite('result/other2/' + filename, img)


def classify_by_lines(img_path, filename, lines):
    img = cv.imread(img_path)

    if lines <= 38:
        cv.imwrite('result/text/' + filename, img)
    elif 40 <= lines <= 45:
        cv.imwrite('result/text/' + filename, img)
    elif lines >= 220:
        cv.imwrite('result/text/' + filename, img)
    else:
        cv.imwrite('result/should_dia/' + filename, img)


def classify_to_dia(img_path, filename, percentage, saturation, contains, lines, contours):

    img = cv.imread(img_path)

    if percentage > 80:
        return 0

    if saturation > 3:
        cv.imwrite('result/image/' + filename, img)
        return 0

    if contains is True:
        if contours > 2600 or 180 < contours < 350:
            cv.imwrite('result/text/' + filename, img)
            return 0
        else:
            cv.imwrite('result/diagram/' + filename, img)
            return 0
    
    if lines <= 38:
        cv.imwrite('result/text/' + filename, img)
    elif 40 <= lines <= 45:
        cv.imwrite('result/text/' + filename, img)
    elif lines >= 220:
        cv.imwrite('result/text/' + filename, img)
    else:
        if contours > 1300:
            cv.imwrite('result/text/' + filename, img)
        elif 180 <= contours <= 206:
            cv.imwrite('result/text/' + filename, img)
        else:
            cv.imwrite('result/diagram/' + filename, img)


        # cv.imwrite('result/diagram/' + filename, img)

def main():

    # img_path = 'test/test-010.jpg'

    # arr = []

    dir = 'test'
    for filename in os.listdir(dir):
        img_path = os.path.join(dir, filename)

        percentage = thresholding(img_path)
        saturation = calc_average_saturation(img_path)
        contains = contain_color(img_path)
        lines = hough_line_p(img_path)
        contours = find_contours(img_path)




        # classify(img_path, filename, contains)

        # classify_by_lines(img_path, filename, lines)

        classify_to_dia(img_path, filename, percentage, saturation, contains, lines, contours)


        
    #     arr.append(contours)
    
    # arr.sort()
    
    # print("For dia")
    # for i in arr:
    #     print(i)


if __name__ == "__main__":
    main()
