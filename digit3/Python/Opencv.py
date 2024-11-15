import cv2

# 흑백 이미지 출력 (imread flags 활용)

# 이미지 읽기 함수 imread(filename, flags)
# filename : 파일명
# flag
#  - cv2.IMREAD_UNCHANGED : 파일 속성 그대로 읽기
#  - cv2.IMREAD_GRAYSCALE : Graysacle 로 읽기
image = cv2.imread("C:/Python/image.jpg", cv2.IMREAD_UNCHANGED)
gray_image = cv2.imread("C:/Python/image.jpg", cv2.IMREAD_GRAYSCALE)

# 이미지 화면 출력 함수 imshow(winname, filename)
# winname : 이미지 창 이름
# filename : 파일명
cv2.imshow("image", image)
cv2.imshow("image", gray_image)

# waitKey() : 키 입력을 기다리는 대기 함수
cv2.waitKey()

# destroyAllWindows() : 모든 창을 닫기
cv2.destroyAllWindows()

#-------------------------------------------------------------------

# 흑백 이미지 출력 (색상 공간 변환)

src = cv2.imread("C:/Python/image.jpg", cv2.IMREAD_UNCHANGED)

# cvtColor(src, code, desCn) : 이미지의 색상 공간을 변경
# src : 입력 이미지
# code : 변환하려는 색상 공간을 지정하는 플래그
# desCn : 출력 이미지의 채널 수를 지정
dst = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

cv2.imshow("src", src)
cv2.imshow("dst", dst)
cv2.waitKey()
cv2.destroyAllWindows()

#-------------------------------------------------------------------

# 이미지 자르기

# 자르기(Slice)는 영상이나 이미지에서 특정 영역을 잘라내는 연산을 의미
# 특정 영역을 잘라내는 것을 관심 영역(ROI)
# 관심 영역(ROI)은 이미지 상에서 관심 있는 영역을 의미

src = cv2.imread("C:/Python/image.jpg", cv2.IMREAD_COLOR)

# src 이미지에 src[행, 열]로 관심 영역을 설정
dst = src[0:1080, 0:1920]

cv2.imshow("src", src)
cv2.imshow("dst", dst)
cv2.waitKey()
cv2.destroyAllWindows()

#-------------------------------------------------------------------

# 이진화(Binary)

# 어느 지점을 기준으로 값이 높거나 낮은 픽셀의 값을 대상으로 특정 연산을 수행할 때 사용
# 일반적으로 값이 높거나 낮은 픽셀을 검은색 또는 흰색의 값으로 변경

src = cv2.imread("C:/Python/image.jpg", cv2.IMREAD_COLOR)

gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

# threshold(src, thresh, maxval, type) : 입력 이미지(src)를 임곗값 형식(type)에 따라 
# 임곗값(thresh)과 최댓값(maxval)을 활용하여 설정 임곗값(ret)과 결과 이미지(dst)를 반환
ret, dst = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

# cv2.THRESH_BINARY : dst = (src > thresh) ? maxval : 0
# 임곗값을 초과할 경우 maxval, 아닐 경우 0

cv2.imshow("dst", dst)
cv2.waitKey()
cv2.destroyAllWindows()