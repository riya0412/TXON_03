import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import csv

path = 'ImageAttendence'
images=[]
classNames=[]
myList=os.listdir(path)
print(myList)
for cl in myList:
    curImg=cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodeList=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def AttendMark(name):
    with open("Attendence.csv","r+",newline='') as f1:
        myRecord=f1.readlines()
        nameList=[]
        for line in myRecord:
            Entry=line.split(',')
            nameList.append(Entry[0])
        writer=csv.writer(f1)
        if name not in nameList:
            nowtime= datetime.now()
            dtstr=nowtime.strftime("%H:%M:%S")
            writer.writerow([name,dtstr])            

encodeListKnown=findEncodings(images)
print("Encoding Complete")

cap=cv2.VideoCapture(0)

while True:
    success, img=cap.read()
    imgSmall=cv2.resize(img,(0,0),None,0.25,0.25)
    imgSmall=cv2.cvtColor(imgSmall,cv2.COLOR_BGR2RGB)
    facesCurFrame=face_recognition.face_locations(imgSmall)
    encodeCurFrame=face_recognition.face_encodings(imgSmall,facesCurFrame)

    for encodeface,faceloc in zip(encodeCurFrame,facesCurFrame):
        matches= face_recognition.compare_faces(encodeListKnown,encodeface)
        faceDis=face_recognition.face_distance(encodeListKnown,encodeface)
        # print(faceDis)
        matchIndex=np.argmin(faceDis)

        if matches[matchIndex]:
            name=classNames[matchIndex].upper()
            # print(name)
            y1,x2,y2,x1=faceloc
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_DUPLEX,1,(255,255,255),2)
            AttendMark(name)


    cv2.imshow('webcam',img)
    cv2.waitKey(1)



