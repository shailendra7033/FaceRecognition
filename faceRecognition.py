import cv2
import os
import numpy as np

def faceDetection(test_img):
    gray_img=cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
    face_haar_cascade=cv2.CascadeClassifier("D:\Engage'22\FaceRecognition\haarcascade_frontalface_default.xml")
    faces =face_haar_cascade.detectMultiScale(gray_img,scaleFactor=1.32,minNeighbors=5)
    # faces=face_haar_cascade.detectMultiScale(test_img,scaleFactor=1.32,minNeighbors=5)

    return faces,gray_img



# function for labels_for_training data

def labels_for_training_data(directory):
    faces=[]
    faceID=[]
    print(directory)
    for path,subdirnames,filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith("."):
                print("skipping system files")
                continue
            id =os.path.basename(path)
            img_path1=os.path.join(path,filename)
            img_path=""
            for i in range(0,len(img_path1)):
                if img_path1[i]=="\\":
                    img_path+="/"
                else:
                    img_path+=img_path1[i]




                
            print("img_path",img_path)
            print("id : ",id)
            test_img=cv2.imread(img_path)
            if test_img is None:
                print("image not loaded properly means cant read")
                continue
            faces_rect,gray_img=faceDetection(test_img)
            if(len(faces_rect)!=1):
                continue
            (x,y,w,h)=faces_rect[0]
            roi_gray=gray_img[y:y+w,x:x+h]
            faces.append(roi_gray)
            faceID.append(int(id))

    return faces,faceID


# to train our classifier

def train_classifier(faces,faceID):
    # print("hii train",faceID)
    face_recognizer=cv2.face.LBPHFaceRecognizer_create()
    # print(faceID)
    face_recognizer.train(faces,np.array(faceID))
    return face_recognizer

#draw rectangle around faces

def draw_rect(test_img,face):
    (x,y,w,h)=face
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(0,255,0),thickness=5)

# to put text on image
def put_text(test_img,text,x,y):
    cv2.putText(test_img,text,(x,y),cv2.FONT_HERSHEY_DUPLEX,2,(0,0,0),2)

