import cv2 as cv
import numpy as np
import os


def calc_average_saturation(img_path):
    image = cv.imread(img_path)

    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    saturation_channel = hsv_image[:, :, 1]

    average_saturation = np.mean(saturation_channel)

    return average_saturation


def harris_corners(img_path):
    img = cv.imread(img_path)
    img = cv.GaussianBlur(img, (3, 3), 3)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv.cornerHarris(gray, 2, 3, 0.04)

    dst = cv.dilate(dst, None)

    img[dst > 0.01 * dst.max()] = [0, 0, 255]

    # cv.imshow('dst', img)
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()

    return len(dst)


def hough_lines(img_path):
    img = cv.imread(img_path)
    img = cv.resize(img, (500, 800))
    img = cv.GaussianBlur(img, (3, 3), cv.BORDER_DEFAULT)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    edges = cv.Canny(gray, 50, 150, apertureSize=3)

    # thresh_val = 130
    # _, bin_img = cv.threshold(gray, thresh_val, 255, cv.THRESH_BINARY)

    lines = cv.HoughLines(edges, 1, np.pi/180, 200)

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
    img = cv.resize(img, (500, 800))
    img = cv.GaussianBlur(img, (5, 5), 5)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    edges = cv.Canny(gray, 50, 150, apertureSize=3)

    contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    cv.drawContours(img, contours, -1, (255,0,0), 1)

    # cv.imshow('contours', img)

    cv.waitKey(0)
    cv.destroyAllWindows()

    return len(contours)


def classify_image_and_others(img_path, filename, saturation):
    img = cv.imread(img_path)

    if saturation > 3:
        cv.imwrite('result/image/' + filename, img)
    else:
        cv.imwrite('result/text_and_diagram/' + filename, img)


def thresholding(img_path):
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    img = cv.resize(img, (500, 800))

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


def main():

    img_path = 'test/test-268.jpg'

    # thresholding(img_path)


    # corners = harris_corners(img_path)
    # print("corners: ", corners)

    # lines = hough_lines(img_path)
    # print("lines: ", lines)

    # cont = find_contours(img_path)
    # print("contours: ", cont)

    # percentage = thresholding(img_path)
    # print(f'black pix: {percentage} %')

    # dir = 'test'
    # for filename in os.listdir(dir):
    #     img_path = os.path.join(dir, filename)

    #     saturation = calc_average_saturation(img_path)
    #     contours = find_contours(img_path)
    #     percentage = thresholding(img_path)

    #     classify_images(img_path, filename, saturation, contours, percentage)

        # percentage = thresholding(img_path)
        # classify(img_path, filename, percentage)

        # average_saturation = calc_average_saturation(img_path)
        # classify_image_and_others(img_path, filename, average_saturation)

        

if __name__ == "__main__":
    main()
