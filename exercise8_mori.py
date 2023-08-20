import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt



def main():
    img_paths = []

    dir = 'test'
    for filename in os.listdir(dir):
        img_path = os.path.join(dir, filename)
        img_paths.append(img_path)


if __name__ == "__main__":
    main()
