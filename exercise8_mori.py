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


def harris_corners(img_path, window_size, k, threshold):
    img = cv.imread(img_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    img_gaussian = cv.GaussianBlur(gray, (3, 3), 0)

    height = img.shape[0]
    width = img.shape[1]

    matrix_R = np.zeros((height, width))

    # Calculate the x e y image derivatives
    dx = cv.Sobel(img_gaussian, cv.CV_64F, 1, 0, ksize=3)
    dy = cv.Sobel(img_gaussian, cv.CV_64F, 0, 1, ksize=3)

    # Calculate product and second derivatives
    dx2 = np.square(dx)
    dy2 = np.square(dy)
    dxy = dx * dy

    offset = int(window_size / 2)

    # Calculate the sum of the products of derivatives for each pixel
    print ("Finding Corners...")
    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            Sx2 = np.sum(dx2[y - offset : y + 1 + offset, x - offset : x + 1 + offset])
            Sy2 = np.sum(dy2[y - offset : y + 1 + offset, x - offset : x + 1 + offset])
            Sxy = np.sum(dxy[y - offset : y + 1 + offset, x - offset : x + 1 + offset])

            # Define the matrix H(x, y)=[ [Sx2, Sxy], [Sxy, Sy2] ]
            H = np.array([[Sx2, Sxy], [Sxy, Sy2]])

            # Calculate the response function ( R=det(H)-k(Trace(H))^2 )
            det = np.linalg.det(H)
            tr = np.matrix.trace(H)
            R = det-k*(tr**2)
            matrix_R[y - offset, x - offset] = R

    # Apply a threshold
    cv.normalize(matrix_R, matrix_R, 0, 1, cv.NORM_MINMAX)

    corners = 0

    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            value=  matrix_R[y, x]
            if value > threshold:
                # cornerList.append([x, y, value])
                cv.circle(img, (x, y), 3, (0, 0, 255))

                corners += 1
    
    print(corners)

    plt.figure("Manually implemented Harris detector")
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)), plt.title("Manually implemented Harris detector")
    plt.xticks([]), plt.yticks([])
    plt.show()


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

    img_path = 'test/test-210.jpg'

    # thresholding(img_path)


    corners = harris_corners(img_path, 5, 0.04, 0.30)
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
