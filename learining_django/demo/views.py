from django.shortcuts import render
import cv2
import os
import numpy as np
from django.http import HttpResponse
# Create your views here.
def index(request):
    number=[0,1,2,3,4,5]
    context={
        'number':number,
    }
    

    def dataset_generator():
        face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        def face_cropped(img):
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray,1.3,5)
            #scalig factor = 1.3
            #minm. neighbour = 5
            if faces == ():
                return None
            for (x,y,w,h) in faces:
                cropped_faces = img[y:y+h,x:x+w]
            return cropped_faces
        
        user_name = input("Enter the name of the user")
        dir_path = os.path.join("dataset./", user_name)
        new_dir = os.mkdir(dir_path)
        capture = cv2.VideoCapture(0)
        id = 0

        while True:
            ret, frame = capture.read()
            if face_cropped(frame) is not None:
                id += 1
                face = cv2.resize(face_cropped(frame),(500,500))
                gray = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
                file_name_path =f"dataset./{user_name}/"+str(user_name)+str(id)+".jpg"
                cv2.imwrite(file_name_path,gray)

                cv2.putText(face, str(id), (50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,250,0),2)

                cv2.imshow("image collector",face)

                if cv2.waitKey(1)==13 or int(id)==200:
                    break

        capture.release()
        cv2.destroyAllWindows()
        dataset_generator()
    return render(request,'demo/index.html',context)