import cv2 as cv
import numpy as np
import os

def detect_edge(img_path):
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

    resized_img = cv.resize(img, (400, 600))

    blur = cv.GaussianBlur(resized_img, (3, 3), 3)

    detected_edge = cv.Canny(blur, threshold1=200, threshold2=250, L2gradient=True)

    return detected_edge

    # cv.imshow('Detected edge', edetected_edge)

    # cv.waitKey(0)
    # cv.destroyAllWindows()


def calc_edge_density(edge_detected):
    edge_pixel_count = np.count_nonzero(edge_detected == 255)
    
    total_pixels = edge_detected.size
    
    edge_density = edge_pixel_count / total_pixels

    return edge_density


def find_contour(edge_detected):
    contours, hierarchy = cv.findContours(edge_detected, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    return contours, hierarchy

    # contour_img = cv.cvtColor(edge_detected, cv.COLOR_GRAY2BGR)
    # cv.drawContours(contour_img, contours, -1, (0, 255, 0), 1)

    # cv.imshow('found contours', contour_img)

    # cv.waitKey(0)
    # cv.destroyAllWindows()


def main():

    dir = 'test'

    for file in os.listdir(dir):
        img_path = os.path.join(dir, file)

        edge_detected = detect_edge(img_path)

        edge_density = calc_edge_density(edge_detected)

        contour_and_hierarchy = find_contour(edge_detected)


if __name__ == "__main__":
    main()
