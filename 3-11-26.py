import cv2
import numpy

flame = cv2.imread("flame.png", cv2.IMREAD_COLOR)
gray = cv2.cvtColor(flame, cv2.COLOR_BGR2GRAY)
blur = cv2.blur(gray,(5,5))

circle = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 20, param1= 50, param2=30, minRadius=1, maxRadius=40)
cv2.imshow("Circle Detection", circle)
cv2.waitKey(0)
cv2.destroyAllWindows()