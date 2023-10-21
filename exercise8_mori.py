import cv2 as cv
import numpy as np
import os
import fitz


def extract_pdf(pdf_path, i):
    pdf = fitz.open(pdf_path)

    for page_num in range(pdf.page_count):
        page = pdf.load_page(page_num)    
        image = page.get_pixmap()
        image.save(f"resource/{i}/{page_num + 1}.jpg") # change the output path later


def create_dir(i):

    result_arr = ['diagram', 'text', 'image']
    parent_dir = './'

    # create each result directory
    for dir_name in result_arr:
        result_dia = f'result/{i}/{dir_name}'
        result_path = os.path.join(parent_dir, result_dia)
        
        if not os.path.exists(result_path):
            os.makedirs(result_path)

    # create each resource directory
    resource_dir = f'resource/{i}'
    resource_path = os.path.join(parent_dir, resource_dir)

    if not os.path.exists(resource_path):
        os.makedirs(resource_path)


def image_saturation(img_path):
    img = cv.imread(img_path)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    saturation = hsv[:, :, 1].mean()

    return saturation


def contain_color(img_path):
    img = cv.imread(img_path)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])

    lower_red = np.array([0, 50, 50])
    upper_red = np.array([20, 255, 255])

    blue_mask = cv.inRange(hsv, lower_blue, upper_blue)
    red_mask = cv.inRange(hsv, lower_red, upper_red)

    blue_pixels_present = cv.countNonZero(blue_mask)
    red_pixels_present = cv.countNonZero(red_mask)

    if blue_pixels_present > 90 or red_pixels_present > 90:
        return True # diagram
    else:
        return False # NOT diagram


def hough_line_p(img_path, filename):
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

    if img is None:
        print ('Error opening image!')
        return -1
    
    edges = cv.Canny(img, 50, 150)
    # cv.imwrite(f'canny/{filename}', edges)

    c_edges = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)

    # returns (x0, y0, x1, y1)

    linesP = cv.HoughLinesP(edges, 1, np.pi / 180, 250, None, 60, 5)

    grad = gradient_of_line(linesP)

    # print(grad)

    lines = np.count_nonzero(grad)

    return lines

    # if linesP is not None:
    #     for i in range(0, len(linesP)):
    #         l = linesP[i][0]
    #         cv.line(c_edges, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
        
    # cv.imwrite(f'lines/{filename}', c_edges)


def gradient_of_line(coordinates):

    # get x0 and y0 coordinates
    x0, y0 = coordinates[:, :, 0], coordinates[:, :, 1]

    # get x1 and y1 coordinates
    x1, y1 = coordinates[:, :, 2], coordinates[:, :, 3]

    # Check if the denominator (x1 - x0) is zero
    denominator_zero = (x1 - x0) == 0

    # Calculate gradient, but set the value to 0 when denominator is zero
    gradient = (y1 - y0) // (x1 - x0)
    gradient[denominator_zero] = 0

    return gradient


def classify(img_path, filename, saturation, contains, lines, dir_num):
    img = cv.imread(img_path)

    if saturation > 2:
        cv.imwrite(f'result/{dir_num}/image/{filename}', img)
        return 0
    
    if contains is True:
        cv.imwrite(f'result/{dir_num}/diagram/{filename}', img)
    else:

        if lines < 3:
            cv.imwrite(f'result/{dir_num}/text/{filename}', img)
        else:
            cv.imwrite(f'result/{dir_num}/diagram/{filename}', img)


# 線の長さを検出してみる(maxとminを両方使ってやってみる)
# xとyの座標の位置がどの象限に入るかにやってみる


def main():

    # add path to pdf to be extracted
    for i in range(1, 23):
        create_dir(i)

        pdf_path = f'pdfs_image_classification_task/pdfs/{i}.pdf'
        extract_pdf(pdf_path, i)
        print(f'{i}.pdf extracted successfully!')


    dir = f'' # directory path to be classified 
    print(dir)
    for filename in os.listdir(dir):
        img_path = os.path.join(dir, filename)

        sat = image_saturation(img_path)
        contains = contain_color(img_path)
        lines = hough_line_p(img_path, filename)
        
        classify(img_path, filename, sat, contains, lines)


if __name__ == "__main__":
    main()