import cv2
import numpy as np

image = cv2.imread('images/bouquet.jpg') # 이미지 읽기

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # BGR에서 HSV 색상 공간으로 변환

# 색상 범위 정의 (HSV) 예시)노란색 
lower_color = np.array([20, 100, 100])  # 색상 하한값 (채도와 명도도 설정)
upper_color = np.array([30, 255, 255])  # 색상 상한값

mask = cv2.inRange(hsv, lower_color, upper_color) # 색상 범위에 해당하는 마스크 생성

result = cv2.bitwise_and(image, image, mask=mask) # 원본 이미지에 마스크 적용 (노란색만 남기고 나머지는 검은색)

# 결과 표시
cv2.imshow('extraction_result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 결과 이미지 저장
cv2.imwrite('extraction_result.jpg', result)



