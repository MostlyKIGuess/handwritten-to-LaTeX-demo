import cv2
import numpy as np


image = cv2.imread("test.png")


resized_image = cv2.resize(image, (45, 45), interpolation=cv2.INTER_AREA)


if len(resized_image.shape) == 3:
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
else:
    gray_image = resized_image

_, bw_image = cv2.threshold(gray_image, 225, 256, cv2.THRESH_BINARY)


cv2.imwrite("test_actual.jpg", bw_image)