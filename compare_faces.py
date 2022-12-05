import cv2
import face_recognition
import math

imgRonaldo = face_recognition.load_image_file('Images/Ronaldo.jpg')
imgRonaldo = cv2.cvtColor(imgRonaldo, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('Images/Ronaldo-new.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgRonaldo)[0]
encodeRonaldo = face_recognition.face_encodings(imgRonaldo)[0]
cv2.rectangle(imgRonaldo,(faceLoc[3], faceLoc[0]),(faceLoc[1], faceLoc[2]), (255,0,255,0),2)


faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]),(faceLocTest[1], faceLocTest[2]), (255,0,255,0),2)

results = face_recognition.compare_faces([encodeRonaldo], encodeTest)
faceDis = face_recognition.face_distance([encodeRonaldo], encodeTest)
print(results, faceDis)
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

def face_distance_to_conf(face_distance, face_match_threshold=0.6):
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    else:


        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))

cv2.imshow("Ronaldo", imgRonaldo)
cv2.imshow("Ronaldo Test", imgTest)
cv2.waitKey(0)