import cv2
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer.create()
face_cascade = cv2.CascadeClassifier('haar_cascade.xml')

impath = 'WIN_20250306_15_58_43_Pro.jpg'

label = 0

faces = []
labels = []

img = cv2.imread(impath)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces_rec = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
if len(faces_rec) == 0:
    print('no faces detected')
else:
    for (x,y,w,h) in faces_rec:
        face = gray[y:y+h, x:x+w]
        faces.append(face)
        labels.append(label)

recognizer.train(faces, np.array(labels))

recognizer.save('trained.yml')

print('Training done')



