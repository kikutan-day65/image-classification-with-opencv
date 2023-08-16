import cv2 as cv


def show_canny(img_path):
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

    resized_img = cv.resize(img, (400, 600))

    blur = cv.GaussianBlur(resized_img, (3, 3), 3)

    edge_detection = cv.Canny(blur, threshold1=100, threshold2=200, L2gradient=True)

    cv.imshow('Detected edge', edge_detection)

    cv.waitKey(0)
    cv.destroyAllWindows()


def main():
    img_path = 'test/test-164.jpg'

    show_canny(img_path)


if __name__ == "__main__":
    main()
