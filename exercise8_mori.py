import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans


def main():
    img_path = []

    dir = 'test'
    for filename in os.listdir(dir):
        path = os.path.join(dir, filename)
        img_path.append(path)



if __name__ == "__main__":
    main()
