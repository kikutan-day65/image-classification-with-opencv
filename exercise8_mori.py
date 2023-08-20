import cv2 as cv
import numpy as np
import os


def detect_edges(img_path):
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

    blur = cv.GaussianBlur(img, (3, 3), 3)

    edge_detection = cv.Canny(blur, threshold1=100, threshold2=200, L2gradient=True)

    return edge_detection


def find_contours(edge_detected):
    contours, hierarchies = cv.findContours(edge_detected, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    return contours


def calc_average_saturation(img_path):
    image = cv.imread(img_path)

    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    saturation_channel = hsv_image[:, :, 1]

    average_saturation = np.mean(saturation_channel)

    return average_saturation


def classify_image_and_others(img_path, filename, saturation):
    img = cv.imread(img_path)

    if saturation > 5:
        cv.imwrite('result/image/' + filename, img)
    else:
        pass


def main():
    dir = 'test'
    for filename in os.listdir(dir):
        img_path = os.path.join(dir, filename)
            
        average_saturation = calc_average_saturation(img_path)

        classify_image_and_others(img_path, filename, average_saturation)
    

if __name__ == "__main__":
    main()