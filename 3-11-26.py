import cv2
import numpy as np

flame = cv2.imread("flame.jpg")
gray=cv2.cvtColor(flame,cv2.COLOR_BGR2GRAY)
blur = cv2.medianBlur(gray,5)

circlesDetected = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 20, param1= 50, param2=30, minRadius=1, maxRadius=70)

if circlesDetected is not None:
    circlesDetected=np.uint16(np.around(circlesDetected))
    for pt in circlesDetected[0,:]:
        a, b ,r = pt[0], pt[1], pt[2]
        circle = cv2.circle(flame, (a,b), r, (0,0,255), 3)
        circle = cv2.circle(flame, (a,b), 1, (0,0,255), 3)

params = cv2.SimpleBlobDetector_Params()
params.filterByArea=True
params.minArea=100
params.filterByCircularity=True
params.minCircularity=0.9
params.filterByConvexity=True
params.minConvexity=0.2
params.filterByInertia=True
params.minInertiaRatio=0.01

detector=cv2.SimpleBlobDetector_create(params)
keypoints=detector.detect(flame)
blank=np.zeros((1,1))
blobs=cv2.drawKeypoints(flame,keypoints,blank,(0,0,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

numberOfBlobs=len(keypoints)
text="Number of Circular Blobs: "+ str(numberOfBlobs)
cv2.putText(blobs,text,(20,550),cv2.FONT_HERSHEY_COMPLEX,1,(0,100, 255),2)


cv2.imshow("Circle",circle)
cv2.imshow("Circular Blobs", blobs)
cv2.waitKey(0)
cv2.destroyAllWindows()
