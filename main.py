import cv2
import face_recognition
from simplefacerec import SimpleFacerec

cap = cv2.VideoCapture("file")

while True:
    ret, frame = cap.read()
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)

    if key == 27: #touche esc
        break

cap.release()
cv2.destroyAllWindows() #closes video file or capturing device (webcam)
