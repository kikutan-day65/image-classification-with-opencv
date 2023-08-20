import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

def detect_edges(img_path):
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

    blur = cv.GaussianBlur(img, (3, 3), 3)

    edge_detection = cv.Canny(blur, threshold1=100, threshold2=200, L2gradient=True)

    return edge_detection


def main():
    # img_paths = []

    # dir = 'test'
    # for filename in os.listdir(dir):
    #     img_path = os.path.join(dir, filename)
    #     img_paths.append(img_path)

    
    img_path = 'data_collection_img/image.jpg'
    
    edges_detected = detect_edges(img_path)


if __name__ == "__main__":
    main()
