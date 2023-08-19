import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

# def detect_edge(img_path):
#     img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

#     resized_img = cv.resize(img, (400, 600))

#     blur = cv.GaussianBlur(resized_img, (3, 3), 3)

#     detected_edge = cv.Canny(blur, threshold1=125, threshold2=175, L2gradient=True)

#     return detected_edge

#     # cv.imshow('Detected edge', detected_edge)

#     # cv.waitKey(0)
#     # cv.destroyAllWindows()


# def calc_edge_density(edge_detected):
#     edge_pixel_count = np.count_nonzero(edge_detected == 255)
    
#     total_pixels = edge_detected.size
    
#     edge_density = edge_pixel_count / total_pixels

#     return edge_density


# def find_contour(edge_detected):
#     contours, hierarchies = cv.findContours(edge_detected, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

#     return contours, hierarchies

#     # contour_img = cv.cvtColor(edge_detected, cv.COLOR_GRAY2BGR)
#     # cv.drawContours(contour_img, contours, -1, (0, 255, 0), 1)

#     # cv.imshow('found contours', contour_img)

#     # cv.waitKey(0)
#     # cv.destroyAllWindows()


def harris_corner_detection(img_path):
    img = cv.imread(img_path)

    # remove image noises
    blur = cv.GaussianBlur(img, (3, 3), 0)

    gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv.cornerHarris(gray, 2, 3, 0.02)

    dst = cv.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst > 0.01 * dst.max()] = [0, 255, 0]

    # cv.imshow('dst', dst)

    cv.waitKey(0)
    cv.destroyAllWindows()


def tomasi_corner_detection(img_path):
    img = cv.imread(img_path)

    # remove image noises
    blur = cv.GaussianBlur(img, (3, 3), 0)

    gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)

    corners = cv.goodFeaturesToTrack(gray, 100, 0.01, 10)
    corners = np.int0(corners)

    for i in corners:
        x, y = i.ravel()
        cv.circle(img, (x, y), 3, 255, -1)
    
    cv.imwrite('./test.jpg', img)


def main():
    img_path = 'test/test-128.jpg'

    harris_corner_detection(img_path)

    tomasi_corner_detection(img_path)


    # dir = 'test'

    # for file in os.listdir(dir):
    #     img_path = os.path.join(dir, file)

    #     # edge_detected = detect_edge(img_path)

    #     # edge_density = calc_edge_density(edge_detected)

    #     # contour_and_hierarchy = find_contour(edge_detected)


if __name__ == "__main__":
    main()
