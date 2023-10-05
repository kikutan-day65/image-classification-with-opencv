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


def hough_line_p(img_path):
    img = crop_image(img_path)

    if img is None:
        print ('Error opening image!')
        return -1

    dst = cv.Canny(img, 200, 400, None, 3)

    cdstP = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)

    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 200, None, 100, 10)

    if linesP is None:
        return 0
    
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
    
    cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
    
    cv.waitKey(0)
    cv.destroyAllWindows()

    return len(linesP)


def main():

    img_path = 'dia/test-244.jpg'

    # dir = 'dia'
    # for filename in os.listdir(dir):
    #     img_path = os.path.join(dir, filename)

    #     matching(img_path, filename)

if __name__ == "__main__":
    main()
