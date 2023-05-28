import cv2
import imutils
import numpy as np
from imutils.object_detection import non_max_suppression

image = cv2.imread("people.jpg")
image = imutils.resize(image, width=1280, height=720)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

rects, weights = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)

rect = np.array([[x, y, x+w, y+h] for x, y, w, h in rects])
picks = non_max_suppression(rect, probs=None, overlapThresh=0.65)

counter = 0
for x1, y1, x2, y2 in picks:
    text = "Human {}".format(counter)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, text, (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    counter += 1

cv2.imshow("Image", image)

cv2.waitKey(0)
cv2.destroyAllWindows()

