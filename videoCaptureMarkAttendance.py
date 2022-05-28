import os
import cv2
import numpy as np
import faceRecognition as fr
import datetime


#This module captures images via webcam and performs face recognition
face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("D:/Engage'22/FaceRecognition/trainingData.yml")
name={4146:"Prem",4183:"Shailendra"}




cap=cv2.VideoCapture(0)
flag=0
while True:
    ret,test_img=cap.read()# captures frame and returns boolean value and captured image
    faces_detected,gray_img=fr.faceDetection(test_img)



    for (x,y,w,h) in faces_detected:
      cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)

    resized_img = cv2.resize(test_img, (1000, 700))
    


    for face in faces_detected:
        (x,y,w,h)=face
        roi_gray=gray_img[y:y+w, x:x+h]
        label,confidence=face_recognizer.predict(roi_gray)#predicting the label of given image
        print("confidence:",confidence)
        print("label:",label)
        fr.draw_rect(test_img,face)
        predicted_name=name[label]
        if confidence <45 and flag==0 :
           flag=1
           fr.put_text(test_img,predicted_name,x,y)
           tod = datetime.date.today()
           nameFile =str(tod)+".txt"
           
           f = open(nameFile, "a")
           nw = datetime.datetime.now().time()
           time_string=str(nw)
           tym=time_string[:8]
        
           f.writelines(predicted_name)
           f.writelines("  Present  ")
           f.writelines(tym)
           f.writelines("\n")
           f.close()
           
           break
        #    cv2.destroyAllWindows
        

        if confidence <45 :
            fr.put_text(test_img,predicted_name,x,y)
        else:
          fr.put_text(test_img,"Unknown Face",x,y)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('face recognition tutorial ',resized_img)
    if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed
        break


cap.release()
cv2.destroyAllWindows
