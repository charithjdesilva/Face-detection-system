import cv2
import numpy as np
import face_recognition

# imgOri = face_recognition.load_image_file('./images/Blake Lively.jpg')    # load training image
imgOri = face_recognition.load_image_file('./images/Emma.jpg')    # load training image
imgOri = cv2.cvtColor(imgOri, cv2.COLOR_BGR2RGB)     # convert to RGB

# imageTest = face_recognition.load_image_file('./images/Blake Lively Test.jpg')    # load testing image
# imageTest = face_recognition.load_image_file('./images/Emma Stone.jpg')    # load testing image
# imageTest = face_recognition.load_image_file('./images/Blake Lively1.jpg')    # load testing image
imageTest = face_recognition.load_image_file('./images/Margot.jpg')    # load testing image
imageTest = cv2.cvtColor(imageTest, cv2.COLOR_BGR2RGB)     # convert to RGB

faceLoc = face_recognition.face_locations(imgOri)[0]  # capture the face location
# [0] is used to access the first element of the list returned by the face_locations(), since we are sending a single image
encodeBlake = face_recognition.face_encodings(imgOri)[0]
cv2.rectangle(imgOri, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255,255,0), 2)    # draw a rectangle on where we detected the page

faceLocTest = face_recognition.face_locations(imageTest)[0]  # capture the face location
# [0] is used to access the first element of the list returned by the face_locations(), since we are sending a single image
encodeTest = face_recognition.face_encodings(imageTest)[0]
cv2.rectangle(imageTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255,255,0), 2)    # draw a rectangle on where we detected the page

results = face_recognition.compare_faces([encodeBlake], encodeTest) # results either true or false
faceDistance = face_recognition.face_distance([encodeBlake], encodeTest) # face_distance() helps us to find out how similar these images are
print(results, faceDistance)

cv2.putText((imageTest), f'{results} {round(faceDistance[0], 2)}', (50,50),cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

cv2.imshow('Blake Original', imgOri)
cv2.imshow('Test', imageTest)
cv2.waitKey(0)

