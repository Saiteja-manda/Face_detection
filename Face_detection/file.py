import cv2

recognizer = cv2.face.LBPHFaceRecognizer.create()
recognizer.read('trained.yml')
 
face_cascade = cv2.CascadeClassifier('haar_cascade.xml')
cap = cv2.VideoCapture(0)

recognized_label = 0
confidence_threshold = 50

while True:
    ret, frame= cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces_rec = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    for (x,y,w,h) in faces_rec:
        face = gray[y:y+h, x:x+w]
        label, confidence=recognizer.predict(face)

        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), thickness=2)

        if label == recognized_label and confidence < confidence_threshold:
          cv2.putText(frame, 'saiteja', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), thickness=2)
        else:
         cv2.putText(frame, 'unknown', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), thickness=2)
        
        cv2.imshow('face_recognition', frame)
    if cv2.waitKey(20) & 0xFF == ord('s'):
        break
cap.release()
cv2.destroyAllWindows