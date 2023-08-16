import cv2 as cv


def show_canny(img_path):
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

    resized_img = cv.resize(img, (700, 1000))

    simple_canny = cv.Canny(resized_img, threshold1=100, threshold2=200)

    blur = cv.GaussianBlur(resized_img, (3, 3), 3)

    blurred_canny = cv.Canny(blur, threshold1=100, threshold2=200)

    cv.imshow('Simple', simple_canny)
    cv.imshow('blurred', blurred_canny)

    cv.waitKey(0)
    cv.destroyAllWindows()


def main():
    img_path = 'test/test-284.jpg'

    show_canny(img_path)


if __name__ == "__main__":
    main()
