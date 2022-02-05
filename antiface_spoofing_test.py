# import the necessary packages
# Code test
from imutils.video import VideoStream
import tensorflow
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from  tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
#
import tkinter as tk
from pygame import mixer
import time
import cv2
from tkinter import *
import tkinter.messagebox
root = Tk()
from PIL import Image, ImageTk

#Khoi tao cua so window va setup dao dien
root.geometry('350x370')
frame = Frame(root, relief=RIDGE, borderwidth=2)
frame.pack(fill=BOTH,expand=1)
root.title('ANTI_SPOOFING_CAM')
frame.config(background='light blue')
label = Label(frame, text="ANTI_SPOOFING_CAM", bg='light blue',font=('Times 20  bold'))
label.pack(side=TOP)
filename = PhotoImage(file="demo.png")
background_label = Label(frame,image=filename)
background_label.pack(side=TOP)
#

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;0"

# Cai dat cac tham so dau vao
ap = argparse.ArgumentParser()
#ap.add_argument("-m", "--model", type=str, default='liveness.model',
#	help="path to trained model")
ap.add_argument("-l", "--le", type=str, default='le.pickle',
	help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

def Exit():
    exit()
#
def open_camera():
   capture =cv2.VideoCapture(0)
   while True:
      ret,frame=capture.read()
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      cv2.imshow('frame',frame)
      if cv2.waitKey(1) & 0xFF ==ord('q'):
         break
   capture.release()
   cv2.destroyAllWindows()
#
def alert():
   mixer.init()
   alert=mixer.Sound('beep-07.wav')
   alert.play()
   time.sleep(0.1)
   alert.play()   
#
def Spoofing():
    print("[INFO] loading face detector...")
    protoPath = r'E:\AI_CUOIKY\face_anti_spoofing-master\deploy.prototxt'
    modelPath = r'E:\AI_CUOIKY\face_anti_spoofing-master\res10_300x300_ssd_iter_140000.caffemodel'
    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # Load model nhan dien fake/real
    print("[INFO] loading liveness detector...")
    model = load_model("test.h5")
    le = pickle.loads(open(args["le"], "rb").read())
   
    #  Doc video tu webcam
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()

    time.sleep(1.0)

    while True:
        # Doc anh tu webcam
        frame = vs.read()

        # Chuyen thanh blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0))

        # Phat hien khuon mat
        net.setInput(blob)
        detections = net.forward()

        # Loop qua cac khuon mat
        for i in range(0, detections.shape[2]):

            confidence = detections[0, 0, i, 2]

            # Neu conf lon hon threshold
            if confidence > args["confidence"]:

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)

                # Lay vung khuon mat
                face = frame[startY:endY, startX:endX]
                face = cv2.resize(face, (32, 32))
                face = face.astype("float") / 255.0
                face = img_to_array(face)
                face = np.expand_dims(face, axis=0)

                # Dua vao model de nhan dien fake/real
                preds = model.predict(face)[0]

                j = np.argmax(preds)
                label = le.classes_[j]

                # Ve hinh chu nhat quanh mat
                label = "{}: {:.4f}".format(label, preds[j])
                if (j==0):
                    # Neu la fake thi ve mau do
                    cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                        (0, 0, 255), 2)
                else:
                    # Neu real thi ve mau xanh
                    cv2.putText(frame, label, (startX, startY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                                (0,  255,0), 2)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # Bam 'q' de thoat
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
   
##
def Spoofing_Sound():
    print("[INFO] loading face detector...")
    protoPath = r'E:\AI_CUOIKY\face_anti_spoofing-master\deploy.prototxt'
    modelPath = r'E:\AI_CUOIKY\face_anti_spoofing-master\res10_300x300_ssd_iter_140000.caffemodel'
    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # Load model nhan dien fake/real
    print("[INFO] loading liveness detector...")
    model = load_model("test.h5")
    le = pickle.loads(open(args["le"], "rb").read())
    #  Doc video tu webcam
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()

    time.sleep(2.0)

    while True:
        # Doc anh tu webcam
        frame = vs.read()

        # Chuyen thanh blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0))

        # Phat hien khuon mat
        net.setInput(blob)
        detections = net.forward()

        # Loop qua cac khuon mat
        for i in range(0, detections.shape[2]):

            confidence = detections[0, 0, i, 2]

            # Neu conf lon hon threshold
            if confidence > args["confidence"]:

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)

                # Lay vung khuon mat
                face = frame[startY:endY, startX:endX]
                face = cv2.resize(face, (32, 32))
                face = face.astype("float") / 255.0
                face = img_to_array(face)
                face = np.expand_dims(face, axis=0)

                # Dua vao model de nhan dien fake/real
                preds = model.predict(face)[0]

                j = np.argmax(preds)
                label = le.classes_[j]

                # Ve hinh chu nhat quanh mat
                label = "{}: {:.4f}".format(label, preds[j])
                if (j==0):
                    # Neu la fake thi ve mau do
                    cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                        (0, 0, 255), 2)
                    alert()
                else:
                    # Neu real thi ve mau xanh
                    cv2.putText(frame, label, (startX, startY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                                (0,  255,0), 2)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # Bam 'q' de thoat
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()

but1=Button(frame,padx= 3 ,pady= 3 ,width= 12 ,bg='white',fg= 'black',relief=GROOVE,text='Open_Camera',command=open_camera,font=('helvetica 14 bold'))
but1.place(x=100,y=50)

but2=Button(frame,padx= 3 ,pady= 3 ,width=12,bg='white',  fg= 'black',relief=GROOVE,text='Anti_Spoofing',command=Spoofing,font=('helvetica 14 bold'))
but2.place(x=100,y=130)

but3=Button(frame,padx= 3 ,pady= 3 ,width= 22,bg='white',  fg= 'black',relief=GROOVE,text='Anti_Spoofing_with_Sound',command= Spoofing_Sound,font=('helvetica 14 bold'))
but3.place(x=34,y=200)

but4=Button(frame,padx= 4 ,pady= 4 ,width= 5,bg='white',  fg= 'black',relief=GROOVE,text='EXIT',command=Exit,font=('helvetica 15 bold'))
but4.place(x=130,y=270)

root.mainloop()

