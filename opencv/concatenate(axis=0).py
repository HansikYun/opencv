import cv2
import numpy as np

image1 = cv2.imread("dog.jpg", cv2.IMREAD_COLOR)
image2 = cv2.imread("cat.jpg", cv2.IMREAD_COLOR)

resized_image1 = cv2.resize(image1, (400, 400))
resized_image2 = cv2.resize(image2, (400, 400))

image = np.concatenate((resized_image1, resized_image2), axis=0)

cv2.imshow("merged image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
