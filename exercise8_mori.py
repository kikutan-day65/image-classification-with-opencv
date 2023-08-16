import cv2 as cv


def detect_edge(img_path):
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

    resized_img = cv.resize(img, (400, 600))

    blur = cv.GaussianBlur(resized_img, (3, 3), 3)

    detected_edge = cv.Canny(blur, threshold1=200, threshold2=250, L2gradient=True)

    return detected_edge

    # cv.imshow('Detected edge', edetected_edge)

    # cv.waitKey(0)
    # cv.destroyAllWindows()


def find_contour(edge_detected):
    contours, hierarchy = cv.findContours(edge_detected, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    return contours

    # contour_img = cv.cvtColor(edge_detected, cv.COLOR_GRAY2BGR)
    # cv.drawContours(contour_img, contours, -1, (0, 255, 0), 1)

    # cv.imshow('found contours', contour_img)

    # cv.waitKey(0)
    # cv.destroyAllWindows()



def main():
    img_path = 'test/test-128.jpg'

    edge_detected = detect_edge(img_path)

    found_contour = find_contour(edge_detected)


if __name__ == "__main__":
    main()
