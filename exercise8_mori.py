import cv2 as cv
import numpy as np
# import matplotlib.pyplot as plt

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


def edge_orientation_histogram(edge_detected):
    gradient_x = cv.Sobel(edge_detected, cv.CV_64F, 1, 0, ksize=3)
    gradient_y = cv.Sobel(edge_detected, cv.CV_64F, 0, 1, ksize=3)

    edge_direction = np.arctan2(gradient_y, gradient_x)

    num_bins = 16  # ヒストグラムのビン数（角度の範囲を均等に分割）
    hist, bins = np.histogram(edge_direction, bins=num_bins, range=(-np.pi, np.pi))

    print(hist)
    print(bins)

    return hist, bins

    # エッジ方向ヒストグラムを表示
    # plt.bar(bins[:-1], hist, width=np.pi/4)
    # plt.xlabel('Edge Direction')
    # plt.ylabel('Frequency')
    # plt.title('Edge Direction Histogram')
    # plt.savefig('edge-orientation-hist.png')

def main():
    img_path = 'test/test-128.jpg'

    edge_detected = detect_edge(img_path)

    edge_density = calc_edge_density(edge_detected)

    contour_and_hierarchy = find_contour(edge_detected)

    edge_orientation_histogram(edge_detected)


if __name__ == "__main__":
    main()
