import cv2 as cv
import numpy as np
import os
import fitz


def extract_pdf(pdf_path):
    pdf = fitz.open(pdf_path)

    for page_num in range(pdf.page_count):
        page = pdf.load_page(page_num)    
        image = page.get_pixmap()
        image.save(f"test/20-{page_num + 1}.jpg") # change the output path later


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

    if blue_pixels_present > 0 or red_pixels_present > 0:
        return True # diagram
    else:
        return False # NOT diagram


def hough_line_p(img_path, filename):
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

    if img is None:
        print ('Error opening image!')
        return -1
    
    edges = cv.Canny(img, 200, 400, None, 3)
    # cv.imwrite(f'canny/{filename}', edges)

    c_edges = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)

    # returns (x0, y0, x1, y1)
    linesP = cv.HoughLinesP(edges, 1, np.pi / 180, 100, None, 100, 10)

    grad = gradient_of_line(linesP)

 
    # if linesP is not None:
    #     for i in range(0, len(linesP)):
    #         l = linesP[i][0]
    #         cv.line(c_edges, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
        
    # cv.imwrite(f'canny/{filename}', c_edges)


def gradient_of_line(coordinates):

    # get x0 and y0 coordinates
    x0, y0 = coordinates[:, :, 0], coordinates[:, :, 1]

    # get x1 and y1 coordinates
    x1, y1 = coordinates[:, :, 2], coordinates[:, :, 3]

    # gradient calculation
    grad = (y1 - y0) // (x1 - x0)

    print(grad)



def classify(img_path, filename, saturation, contains):
    img = cv.imread(img_path)

    if saturation > 2:
        cv.imwrite(f'result/image/{filename}', img)
        return 0
    
    if contains is True:
        cv.imwrite(f'result/diagram/{filename}', img)
        return 0
    
    cv.imwrite(f'result/text/{filename}', img)


def main():

    # add path to pdf to be extracted
    # pdf_path = 'pdfs_image_classification_task/pdfs/20.pdf'
    # extract_pdf(pdf_path)

    # img_path = 'dia/test-248.jpg'
    img_path = 'text/test-198.jpg'

    hough_line_p(img_path, filename='example')

    # arr = []

    # dir = 'dia'
    # for filename in os.listdir(dir):
    #     img_path = os.path.join(dir, filename)

        # sat = image_saturation(img_path)
        # contains = contain_color(img_path)
        # hough_line_p(img_path, filename)

        # classify(img_path, filename, sat, contains)

    # for i in arr:
    #     print(i)


if __name__ == "__main__":
    main()