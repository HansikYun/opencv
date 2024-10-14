import cv2
import numpy as np

image1 = cv2.imread("dog.jpg", cv2.IMREAD_COLOR)

resized_image = cv2.resize(image1, (600, 600))

cv2.imshow("resized image", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
