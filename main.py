import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector

#webcam
video_capture = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)

while True:
   
    ret, frame = video_capture.read()
    
    if not ret:
        break

  
    frame, faces = detector.findFaceMesh(frame, draw=False)

    if faces:
        face = faces[0]
        pointLeft = face[145]
        pointRight = face[374]
        w, _ = detector.findDistance(pointLeft, pointRight)
        W = 6.3
        f = 840
        d = (W * f) / w

        cvzone.putTextRect(frame, f'Depth: {int(d)}cm',
                           (face[10][0] - 100, face[10][1] - 50),
                           scale=2)

   
    cv2.imshow('Video', frame)

   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
