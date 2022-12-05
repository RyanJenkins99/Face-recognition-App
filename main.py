import cv2
import face_recognition as faceRec
import numpy as np
import os
import math

from flask import Flask, render_template, Response
from flask_socketio import SocketIO

# hostname=socket.gethostname()
# IPAddr=socket.gethostbyname(hostname) #Get Ip address
app = Flask(__name__)
socketioApp = SocketIO(app)

path = 'Images'
images = []
Names = []
myList = os.listdir(path)
print(myList)
# print("http://"+IPAddr+":8080")

for c1 in myList:
    curImg = cv2.imread(f'{path}/{c1}')
    images.append(curImg)
    Names.append(os.path.splitext(c1)[0])
print(Names)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = faceRec.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList



encodeListKnown = findEncodings(images)
print('Encoding Complete')



def face_distance_to_conf(face_distance, face_match_threshold=0.6):
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))

cap = cv2.VideoCapture(0)
def gen_frames():
    while True:
        success, img = cap.read()
        imgS = cv2.resize(img,(0,0),None,0.25,0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = faceRec.face_locations(imgS)
        encodesCurFrame = faceRec.face_encodings(imgS,facesCurFrame)

        for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
            matches = faceRec.compare_faces(encodeListKnown,encodeFace)
            faceDis = faceRec.face_distance(encodeListKnown,encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = Names[matchIndex].upper()
                y1,x2,y2,x1 = faceLoc
                y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

            else :
                name = 'Mystery'
                y1,x2,y2,x1 = faceLoc
                y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                
        ret, buffer = cv2.imencode('.jpg', img) # convert capture to jpg for browser
        img = buffer.tobytes()

        # concatenate frames and output
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')
            
        if cv2.waitKey(1) == ord('q'): #break loop if q pressed
                break 

    cap.release() #turn off cam
    cv2.destroyAllWindows() #close all windows

@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    #Video streaming Home Page
    return render_template('index.html')

def run():
    socketioApp.run(app)

if __name__ == '__main__':
    socketioApp.run(app)


    # cv2.imshow('Webcam',img)
    # cv2.waitKey(1)




