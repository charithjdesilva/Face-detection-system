import cv2
import numpy as np
import face_recognition

imgBlake = face_recognition.load_image_file('./images/Blake Lively.jpg')    # load training image
imgBlake = cv2.cvtColor(imgBlake, cv2.COLOR_BGR2RGB)     # convert to RGB

# imageTest = face_recognition.load_image_file('./images/Blake Lively Test.jpg')    # load testing image
# imageTest = face_recognition.load_image_file('./images/Emma Stone.jpg')    # load testing image
imageTest = face_recognition.load_image_file('./images/Blake Lively1.jpg')    # load testing image
imageTest = cv2.cvtColor(imageTest, cv2.COLOR_BGR2RGB)     # convert to RGB

faceLoc = face_recognition.face_locations(imgBlake)[0]  # capture the face location
# [0] is used to access the first element of the list returned by the face_locations(), since we are sending a single image
encodeBlake = face_recognition.face_encodings(imgBlake)[0]
cv2.rectangle(imgBlake, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255,255,0), 2)    # draw a rectangle on where we detected the page

faceLocTest = face_recognition.face_locations(imageTest)[0]  # capture the face location
# [0] is used to access the first element of the list returned by the face_locations(), since we are sending a single image
encodeTest = face_recognition.face_encodings(imageTest)[0]
cv2.rectangle(imageTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255,255,0), 2)    # draw a rectangle on where we detected the page

results = face_recognition.compare_faces([encodeBlake], encodeTest)
print(results)

cv2.imshow('Blake Original', imgBlake)
cv2.imshow('Test', imageTest)
cv2.waitKey(0)

