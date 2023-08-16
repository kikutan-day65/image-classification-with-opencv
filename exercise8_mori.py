import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


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


def main():
    img_path = 'test/test-128.jpg'

    edge_detected = detect_edge(img_path)

    edge_density = calc_edge_density(edge_detected)


if __name__ == "__main__":
    main()
