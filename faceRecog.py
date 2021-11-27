import cv2
import numpy as np
import face_recognition
import os
import imutils
from picamera import PiCamera
import time
from picamera.array import PiRGBArray

path = '.users'
images = []
classNames = []
myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)
print("Total Images : " + str(len(classNames)))


def findEncodings(images):
    x=0;
    print("Encoding...")
    encodeList = []
    for img in images:
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            x+=1
            encodeList.append(encode)
        except:
            print("Encode failed for Image Index : " + str(x) + "\nEncoding Next...")
    print("Total Images Encoded : " + str(len(encodeList)))
    return encodeList

encodeListKnown = findEncodings(images)
print('Encoding Complete')
camera=PiCamera()
camera.resolution=(640,480)
camera.framerate=32
rawCapture=PiRGBArray(camera, size=(640,480))
time.sleep(0.1)

for frame in camera.capture_continuous(
    rawCapture, format='bgr', use_video_port=True):
    img=frame.array
    #success, img = cap.read()

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        if np.any(faceDis<0.45) :
            matchIndex = np.argmin(faceDis)
            if matches[matchIndex]:
                name = classNames[matchIndex]
                print("Welcome "+name)

                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(img, name, (x1 + 6, y2 + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (128, 0, 0), 2)
                print("LED on")
                GPIO.output(21,GPIO.HIGH)    
                
        else :
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            print("No Match Found")
            print("LED off")
            GPIO.output(21,GPIO.LOW)
            
    cv2.imshow('Face Recognition', img)
    cv2.waitKey(1)
    c = cv2.waitKey(1)
    rawCapture.truncate(0)
    if c == 27 or c == 10:
        cv2.destroyAllWindows()
        break