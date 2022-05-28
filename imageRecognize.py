import cv2
import os
from cv2 import resize
import numpy as np
# from faceDetection.faceRecognition import faceDetection
import faceRecognition as fr


# print("hello world")

# this is i didn't get about relative path actual path and difference between  / and  \ in path
test_img=cv2.imread("D:/Engage'22/FaceRecognition/lena.jpg")
# print(test_img)

faces_detected,gray_img=fr.faceDetection(test_img)

# print("face detected:",faces_detected)


faces,faceID=fr.labels_for_training_data("D:\Engage'22\FaceRecognition\images")
# print(faces)
face_recognizer=fr.train_classifier(faces,faceID)
face_recognizer.save('trainingData.yml')

# face_recognizer=cv2.face.LBPHFaceRecognizer_create()
# face_recognizer.read("D:/shailendra_kumar/PYTHON/faceDetection/trainingData.yml")
name={4146:"Prem",4183:"Shailendra"}

for face in faces_detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+h,x:x+h]
    label,confidence=face_recognizer.predict(roi_gray)
    print(confidence)
    if confidence>45:
        fr.put_text(test_img,"Unknown Person",x,y)
        fr.draw_rect(test_img,face)
        print("Not Confident about match")
        continue
    print("label and confidence",label,confidence)
    fr.draw_rect(test_img,face)
    predicted_name=name[label]
    fr.put_text(test_img,predicted_name,x,y)











# for (x,y,w,h) in faces_detected:
#     cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=5)

resize_img=cv2.resize(test_img,(400,400))

cv2.imshow("facedetection",resize_img)

cv2.waitKey(0)
cv2.destroyAllWindows()





